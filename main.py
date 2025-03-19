"""Main script to run the evaluation pipeline."""

import logging
import uuid

import hydra
import mlflow
import pandas as pd
from encourage.llm import BatchInferenceRunner
from encourage.metrics import (
    BLEU,
    F1,
    GLEU,
    ROUGE,
    AnswerSimilarity,
    ContextLength,
    ContextRecall,
    ExactMatch,
    GeneratedAnswerLength,
    MeanReciprocalRank,
    NonAnswerCritic,
    ReferenceAnswerLength,
    AnswerFaithfulness,
    ContextPrecision,
    AnswerRelevance,
)
from encourage.metrics.metric import Metric
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.utils.file_manager import FileManager
from encourage.utils.tracing import enable_mlflow_tracing
from encourage.vector_store import ChromaClient
from hydra.core.config_store import ConfigStore
from vllm import LLM, SamplingParams

from config import Config
from src.data.hf import HuggingFaceDataset
from src.utils.utils import flatten_dict, get_secret

cs = ConfigStore.instance()
cs.store(name="rag-eval", node=Config)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_contexts(df: pd.DataFrame) -> list[Context]:
    """Prepare contexts for the QA dataset."""
    df = df.drop_duplicates(subset=["context_text", "context_id"])
    meta_datas = []
    for i in range(len(df)):
        meta_data = MetaData(tags={"messages": "test_meta_data"})
        meta_datas.append(meta_data)
    return Context.from_documents(
        df["context_text"].tolist(), meta_datas=meta_datas, ids=df["context_id"].tolist()
    )


def init_db(
    collection_name: str,
    contexts: list[Context],
) -> ChromaClient:
    """Initialize the database with the contexts."""
    chroma_client = ChromaClient()
    chroma_client.create_collection(collection_name, overwrite=True)
    chroma_client.insert_documents(collection_name, vector_store_document=contexts)
    return chroma_client


def get_contexts_from_db(
    collection_name: str,
    chroma_client: ChromaClient,
    query_list: list[str],
    top_k: int,
    meta_datas: list[MetaData] = None,
) -> list[Context]:
    """Get the contexts from the database."""
    if meta_datas is not None:
        raise NotImplementedError(
            "Handling of meta_datas is not implemented yet. Look here for more info: https://docs.trychroma.com/guides#querying-a-collection"
        )
    return [
        Context.from_documents(chroma_client.query(collection_name, query, top_k))
        for query in query_list
    ]


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: Config) -> None:
    """Main function to run the evaluation pipeline."""
    enable_mlflow_tracing()

    mlflow.set_tracking_uri(get_secret("uri"))
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    dataset = HuggingFaceDataset(
        cfg.dataset.dataset_path,
        cfg.dataset.dataset_name,
        cfg.dataset.split,
        cfg.dataset.max_samples,
    )
    dataset.raw_data["context_text"] = [ctx[0].get("text") for ctx in dataset.raw_data["ctxs"]]
    unique_values = dataset.raw_data["context_text"].unique()
    uuid_mapping = {val: str(uuid.uuid4()) for val in unique_values}
    dataset.raw_data["context_id"] = dataset.raw_data["context_text"].map(uuid_mapping)

    # Collect last "content" for "user" from each sublist
    last_user_contents = []
    for data in dataset.messages:
        last_user_content = None
        for item in data:
            if item["role"] == "user":
                last_user_content = item["content"]
        last_user_contents.append(last_user_content)

    ## Retrieve the context from the database
    contexts = prepare_contexts(dataset.raw_data)
    chroma_client = init_db(cfg.vector_db.collection_name, contexts)
    contexts = get_contexts_from_db(
        cfg.vector_db.collection_name, chroma_client, last_user_contents, cfg.vector_db.top_k
    )
    meta_datas = []
    for i in range(len(dataset.raw_data)):
        meta_data = MetaData(
            {
                "question": last_user_contents[i],
                "reference_answer": dataset.raw_data["answers"][i][0],
                "reference_document": Document(
                    id=str(dataset.raw_data["context_id"][i]),
                    content=dataset.raw_data["ctxs"][i][0].get("text", ""),
                ),
            }
        )
        meta_datas.append(meta_data)

    ## Create Prompts
    sys_prompts = FileManager(cfg.sys_prompt_path).read()
    prompt_collection = PromptCollection.create_prompts(
        sys_prompts,
        user_prompts=last_user_contents,
        contexts=contexts,
        meta_datas=meta_datas,
        template_name=cfg.template_name,
    )

    ## Init Model and Run Inference
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        top_p=cfg.model.top_p,
    )

    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name,
                                  base_url=cfg.base_url)
    with mlflow.start_run():
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params({"dataset_size": len(dataset.raw_data)})

        with mlflow.start_span(name="root"):
            responses = runner.run(prompt_collection)

        metrics: list[Metric] = [
            GeneratedAnswerLength(),
            ReferenceAnswerLength(),
            ContextLength(),
            MeanReciprocalRank(),
            BLEU(),
            GLEU(),
            ROUGE(rouge_type="rouge1"),
            ROUGE(rouge_type="rouge2"),
            ROUGE(rouge_type="rougeLsum"),
            ExactMatch(),
            F1(),
            AnswerSimilarity(model_name="all-mpnet-base-v2"),
            ContextRecall(runner=runner),
            NonAnswerCritic(runner=runner),
            AnswerFaithfulness(runner=runner),
            ContextPrecision(runner=runner),
            AnswerRelevance(runner=runner),
        ]

        results = {}
        for metric in metrics:
            print(f"Calculate: {metric.name}")
            results[metric.name] = metric(responses)

        print("=" * 30 + "\n" + "Evaluation report" + "\n" + "=" * 30)
        for metric_name, metric_value in results.items():
            print(f"{metric_name}: {metric_value.score}")
            mlflow.log_metric(metric_name, metric_value.score)

        json_dump = [flatten_dict(response.to_dict()) for response in responses.response_data]
        FileManager(cfg.output_file_path).dump_json(json_dump)
        mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file="response.json")


if __name__ == "__main__":
    main()
