import weave
from weave import WeaveList
from weave.flow.scorer import Scorer
from typing import Any, Optional, List

from tqdm import tqdm
import numpy as np

from weave_utils import (
    ChatModel,
    PromptTemplate
)

# TODO: refactor the scorers for the performance, safety, governance overview more intuitive
# TODO: find a way to structure cost and latency (maybe other infos) in eval overview through different scorers and different dicts for now (as discussed with Scott)
# TODO: the prompt argument can actually be backed in statically to the respective class (prompts as public vars at doc top)
# TODO: check out instructor and binary grade implementation from LC implementation
# TODO: feedback for ThirdPartyMetricsScorer base clase from Anish
# - I don't understand the core benefit of having that base class?
# - checking if packages are installed and defining the module string seems to be complicated
# - for a potential llm_judge base class as args (check my past tries, Anish's try and think of beneift for presets marketplace? like this example marketplace)
#    - chat_model      - the model that should be used for the evaluation (ChatModel)
#    - prompt_template - prompt template used by chat model (PromptTemplate)
#    - input parsing   - to be able to aggregate or join as is needed for hallucination
#    - output parsing  - to be able to store the text output from chat model into bool or float
    
#######################
# Performance Metrics #
####################### 
## retrieval scorer ##
@weave.op()
def eval_retrieval(model_output: Optional[dict], main_source: str) -> dict:
    """Evaluate the retrieval accuracy of the predictions: check whether top source document returned by the
       RetrievalQA chain equals the original source document.
       Args:
           - model_output: the dict that will be provided by the model that is evaluated
           - main_source: the target source - as defined in the dataset"""
    
    # post-process prediction results from RetrievalQA in the weave setup.RagNodel
    nr1_retrieval = model_output["source_documents"][0]["url"]
    return nr1_retrieval == main_source
    #return {"first retrieval correct": nr1_retrieval == main_source}

## correctness scorer ##
class CorrectnessLLMJudge(Scorer):
    prompt: PromptTemplate
    models: List[ChatModel]

    @weave.op()
    async def score(self, model_output: Optional[dict], query: str, answer: str, main_source:str, ) -> Any:
        """Score the correctness of the predictions by comparing the query, target answer and pred answer.
           Args:
            - model_output: the dict that will be provided by the model that is evaluated
            - query: the question asked - as defined in the dataset
            - answer: the target answer - as defined in the dataset"""

        # prompt formatting
        human_prompt_args = {
            "query": query,
            "answer": answer,
            "result": model_output["result"],
            }
        messages = self.prompt.format_prompt(
            human_prompt_args=human_prompt_args
        )

        # chat model inference 
        correct_bools = {}
        for model in self.models:
            grade = await model.predict(messages)
            model_name = "-".join(model.chat_model.split("."))
            correct_bools[model_name] = "incorrect" not in grade["content"].strip().lower()

        retrieval_nr1_bool = eval_retrieval(model_output=model_output, main_source=main_source)
        return {"correct": correct_bools, "first_retrieval": retrieval_nr1_bool}
    
    @weave.op()
    def summarize(self, score_rows: WeaveList) -> Optional[dict]:
        """Aggregate all the scores that are calculated for each row by the scoring function.
           Args:
            - score_rows: a WeaveList object, nested dict of metrics and scores
           Returns:
            - nested dict with the same structure as the input"""
        
        # Process correctness data
        correctness_summary = {}
        all_scores = []
        for model_correct_key in tqdm(score_rows[0].get("correct", {}).keys(), desc="Multi-Judge Correctness Calculation"):
            valid_correct_data = [
                row["correct"].get(model_correct_key) 
                for row in score_rows 
                if row.get("correct") and row["correct"].get(model_correct_key) is not None
            ]
            count_true = valid_correct_data.count(True)
            int_data = [int(x) for x in valid_correct_data]
            sample_mean = np.mean(int_data) if int_data else 0

            # Append to all_scores for overall average calculation
            all_scores.extend(valid_correct_data)
            
            correctness_summary[model_correct_key] = sample_mean
            # {
            #     "#": count_true,
            #     "%": sample_mean,
            # }
        
        # Calculate the overall average correctness score if there are any scores
        overall_true = list(all_scores).count(True)/len(correctness_summary)
        overall_int = [int(x) for x in all_scores]
        overall_average = np.mean(overall_int) if overall_int else 0
        correctness_summary["Total Avg"] = overall_average
        # {
        #     "#": overall_true,
        #     "%": overall_average
        # }

        # Process retrieval data
        retrieval_valid_data = [x.get("first_retrieval") for x in score_rows if x.get("first_retrieval") is not None]
        retrieval_count_true = list(retrieval_valid_data).count(True)
        retrieval_int_data = [int(x) for x in retrieval_valid_data]
        retrieval_sample_mean = np.mean(retrieval_int_data) if retrieval_int_data else 0
        return {
            "Correctness": correctness_summary,
            "Nr1_Retrieval": retrieval_sample_mean,
            # {
            #     "#": retrieval_count_true,
            #     "%": retrieval_sample_mean,
            # },
        }


##################
# Safety Metrics #
##################
# TODO: check out different safety measure in Anish's RAG
# TODO: for stuff aggregation - same code as for the RAGModel -> create stuff aggregation as class function to be called here

## hallucination scorer
class HallucinationLLMJudge(Scorer):
    prompt: PromptTemplate
    models: List[ChatModel]

    @weave.op()
    async def score(self, model_output: Optional[dict], query: str) -> Any:
        """Score the hallucination of the predictions by comparing the chat context, query, and result.
           We use "stuff" context aggregation for the chat context.
           Args:
            - model_output: the dict that will be provided by the model that is evaluated
            - query: the question asked - as defined in the dataset"""

        # stuff aggregation
        context_documents = [x["page_content"] for x in model_output["source_documents"]]
        chat_context = "\n\n".join(
            [f"Context {i + 1}:\n{doc}" for i, doc in enumerate(context_documents)]
        )

        # prompt formatting
        human_prompt_args = {
            "chat_context": chat_context,
            "query": query,
            "result": model_output["result"],
        }
        messages = self.prompt.format_prompt(
            human_prompt_args=human_prompt_args
        )

        # chat model inference 
        hallucination_bools = {}
        for model in self.models:
            grade = await model.predict(messages)
            model_name = "-".join(model.chat_model.split("."))
            hallucination_bools[model_name] = "yes" in grade["content"].strip().lower()
        return {"no_hallucination": hallucination_bools}
    
    @weave.op()
    def summarize(self, score_rows: WeaveList) -> Optional[dict]:
        """Aggregate all the scores that are calculated for each row by the scoring function.
           Args:
            - score_rows: a WeaveList object, nested dict of metrics and scores
           Returns:
            - nested dict with the same structure as the input"""

        hallucination_summary = {}
        all_scores = []
        for model_hallucination_key in tqdm(score_rows[0].get("no_hallucination", {}).keys(), desc="Multi-Judge Hallucination Calculation"):
            valid_hallucination_data = [
                row["no_hallucination"].get(model_hallucination_key) 
                for row in score_rows 
                if row.get("no_hallucination") and row["no_hallucination"].get(model_hallucination_key) is not None
            ]
            count_true = valid_hallucination_data.count(True)
            int_data = [int(x) for x in valid_hallucination_data]
            sample_mean = np.mean(int_data) if int_data else 0

            # Append to all_scores for overall average calculation
            all_scores.extend(valid_hallucination_data)
            
            hallucination_summary[model_hallucination_key] = sample_mean
            # {
            #     "#": count_true,
            #     "%": sample_mean,
            # }

        # Calculate the overall average hallucination score if there are any scores
        overall_true = list(all_scores).count(True)/len(hallucination_summary)
        overall_int = [int(x) for x in all_scores]
        overall_average = np.mean(overall_int) if overall_int else 0
        hallucination_summary["Total Avg"] = overall_average
        # {
        #     "#": overall_true,
        #     "%": overall_average
        # }

        return {
            "No Hallucination": hallucination_summary
        }

######################
# Governance Metrics #
######################
# TODO: add different metrics here (cost, latency, utilizaiton, etc)
