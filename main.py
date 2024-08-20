import os
import weave
import asyncio

import os, yaml
from dotenv import load_dotenv

from weave_utils import (
    download_source_docs, 
    gen_data,
    PromptTemplate,
    EmbeddingModel,
    VectorStore,
    ChatModel,
    RagModel
)

from evaluate import (
    eval_retrieval,
    HallucinationLLMJudge,
    CorrectnessLLMJudge
)

def main(config):
    # init weave experiment
    weave.init(config["entity"] + "/" + config["project_name"])

    if config["setup"]:
        # data extraction, object: source table
        download_source_docs(**config)
        
        # dataset generation, object: generated dataset
        gen_model_instance = ChatModel(
            name="GenModel",
            chat_model=config["gen_eval_model"],
            cm_max_new_tokens=config["gm_max_new_tokens"],
            cm_quantize=config["gm_quantize"],
            cm_temperature=config["gm_temperature"],
            inference_batch_size=config["inference_batch_size"],
            device=config["device"],
        )
        
        prompt_template_instance = PromptTemplate(
            system_prompt=config["eval_system_prompt"],
            human_prompt=config["gen_eval_prompt"],
        )
        
        asyncio.run(gen_data(
            gen_model=gen_model_instance,
            prompt_template=prompt_template_instance,
            raw_data_artifact=config["raw_data_artifact"],
            dataset_artifact=config["dataset_artifact"],
            questions_per_chunk=config["questions_per_chunk"],
            max_chunks_considered=config["max_chunks_considered"],
            source_chunk_size=config["source_chunk_size"],
            source_chunk_overlap=config["source_chunk_overlap"],
        ))

    if config["benchmark"]:
        # create RAG Model #
        # TODO: publish doesn't work because indx is not saved (model post doesn't work / save)
        # TODO: is weave.publish(vdb) even necessary?
        chat_model = ChatModel(
            name = "ChatModelRag",
            chat_model = config["chat_model"],
            cm_max_new_tokens = config["cm_max_new_tokens"],
            cm_quantize = config["cm_quantize"],
            cm_temperature = config["cm_temperature"],
            inference_batch_size = config["inference_batch_size"],
            device = config["device"],
        )
        embedding_model = EmbeddingModel(
            embedding_model = config["embedding_model"],
            device = config["device"],
            embedding_model_norm_embed = config["embedding_model_norm_embed"],
        )
        vdb = VectorStore(
            #name = "ClimateVectorStore_NonAsync",
            docs = weave.ref(config["raw_data_artifact"]).get(),
            key = "page_content",
            embedding_model = embedding_model,
            limit = 100,
            chunk_size = config["chunk_size"],
            chunk_overlap = config["chunk_overlap"],
        )
        wf_rag_model = RagModel(
            chat_model = chat_model,
            vector_store = vdb,
            raw_data_artifact = config["raw_data_artifact"],
            rag_prompt_system = config["rag_prompt_system"],
            rag_prompt_user = config["rag_prompt_user"],
            retrieval_chain_type = config["retrieval_chain_type"],
            inference_batch_size = config["inference_batch_size"],
        )

        # create scoring functions #
        # TODO: when specifying a different name attr but with same model class will we see in Weave?
        # TODO: think about offloading the prompt creation to the judges themselves
        eval_models = []
        for idx, model_name in enumerate(config["eval_model"]):
            eval_model = ChatModel(
                name=f"ChatModelEval_{idx}",
                chat_model=model_name,
                cm_max_new_tokens=config["cm_max_new_tokens"],
                cm_quantize=config["cm_quantize"],
                cm_temperature=config["cm_temperature"],
                inference_batch_size=config["inference_batch_size"],
                device=config["device"],
            )
            eval_models.append(eval_model)

        correctness_prompt = PromptTemplate(
            system_prompt=config["eval_system_prompt"],
            human_prompt=config["eval_corr_prompt_user"],
        )
        hallucination_prompt = PromptTemplate(
            system_prompt=config["eval_system_prompt"],
            human_prompt=config["eval_hall_prompt_user"],
        )

        # run evaluation #
        # TODO: restructure scorers to include the categories more intuitively
        evaluation = weave.Evaluation(
            dataset=weave.ref(config["dataset_artifact"]).get(),
            scorers=[
                CorrectnessLLMJudge(
                    name="Performance Metrics",
                    description="Performance metrics incl. Correctness.",
                    models=eval_models,
                    prompt=correctness_prompt,
                ),
                HallucinationLLMJudge(
                    name="Safety Metrics",
                    description="Safety metrics incl. Hallucination.",
                    models=eval_models,
                    prompt=hallucination_prompt,
                )
            ],
        )
        # NOTE: this has to be async in any case (even if scorer and model.predict are not async)
        asyncio.run(evaluation.evaluate(wf_rag_model))

if __name__ == "__main__":
    config = {}
    file_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_path, 'configs/general_config.yaml'), 'r') as file:
        config.update(yaml.safe_load(file))

    if not load_dotenv(os.path.join(file_path, "configs/benchmark.env")):
        print("Environment variables couldn't be found.")

    main(config)
