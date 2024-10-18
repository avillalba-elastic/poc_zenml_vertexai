import logging
from typing import Annotated, Tuple

from datasets import load_dataset, DatasetDict
import evaluate
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_scheduler,
    BertTokenizerFast,
    BertForSequenceClassification,
)
from zenml import pipeline, step
from zenml.config import DockerSettings, ResourceSettings
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import (
    VertexOrchestratorSettings,
)
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import (
    HFDatasetMaterializer,
)
from zenml.integrations.huggingface.materializers.huggingface_tokenizer_materializer import (
    HFTokenizerMaterializer,
)
from zenml.integrations.huggingface.materializers.huggingface_pt_model_materializer import (
    HFPTModelMaterializer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

docker_settings_cuda = DockerSettings(
    parent_image="pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime",
    requirements=[
        "zenml==0.67.0",
        "datasets==3.0.1",
        "transformers==4.45.2",
        "evaluate==0.4.3",
        "scikit-learn==1.5.2",
    ],
)

docker_settings = DockerSettings(replicate_local_python_environment="poetry_export")


def get_device() -> bool:
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@step(
    settings={
        "resources": ResourceSettings(cpu_count=8, memory="32GB", gpu_count=0),
        "docker": docker_settings,
    },
    output_materializers=HFDatasetMaterializer,
)
def load_data() -> Annotated[DatasetDict, "dataset"]:
    dataset = load_dataset("yelp_review_full", trust_remote_code=True)
    return dataset


@step(
    settings={
        "resources": ResourceSettings(cpu_count=8, memory="32GB", gpu_count=0),
        "docker": docker_settings,
    },
    output_materializers=HFTokenizerMaterializer,
)
def load_tokenizer() -> Annotated[BertTokenizerFast, "tokenizer"]:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return tokenizer


@step(
    settings={
        "resources": ResourceSettings(cpu_count=16, memory="64GB", gpu_count=0),
        "docker": docker_settings,
    },
    output_materializers=HFDatasetMaterializer,
)
def tokenize(
    tokenizer: BertTokenizerFast, dataset: DatasetDict
) -> Annotated[DatasetDict, "tokenized_dataset"]:
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


@step(
    settings={
        "resources": ResourceSettings(cpu_count=8, memory="32GB", gpu_count=1),
        "orchestrator": VertexOrchestratorSettings(
            pod_settings={
                "node_selectors": {
                    "cloud.google.com/gke-accelerator": "NVIDIA_TESLA_P4"
                }
            },
        ),
        "docker": docker_settings_cuda,
    },
    output_materializers=HFPTModelMaterializer,
)
def train(
    tokenized_dataset: DatasetDict,
) -> Annotated[BertForSequenceClassification, "model"]:
    small_train_dataset = (
        tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
    )

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    model = BertForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=5
    )
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = get_device()
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}...")
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    logger.info("Training completed.")
    return model


@step(
    settings={
        "resources": ResourceSettings(cpu_count=8, memory="32GB", gpu_count=1),
        "orchestrator": VertexOrchestratorSettings(
            pod_settings={
                "node_selectors": {
                    "cloud.google.com/gke-accelerator": "NVIDIA_TESLA_P4"
                }
            },
        ),
        "docker": docker_settings_cuda,
    }
)
def validate(
    model: BertForSequenceClassification, tokenized_dataset: DatasetDict
) -> Annotated[float, "accuracy"]:
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    eval_dataloader = DataLoader(small_eval_dataset, shuffle=True, batch_size=8)
    metric = evaluate.load("accuracy")
    device = get_device()

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    accuracy = metric.compute()["accuracy"]
    return accuracy


@pipeline
def training_pipeline():
    dataset = load_data()
    tokenizer = load_tokenizer()
    tokenized_dataset = tokenize(tokenizer=tokenizer, dataset=dataset)
    model = train(tokenized_dataset=tokenized_dataset)
    accuracy = validate(model=model, tokenized_dataset=tokenized_dataset)


if __name__ == "__main__":
    training_pipeline()
