from defs import Datapoint, int2str_label, get_misclassified_query_information
from typing import List, Dict, Tuple
from query import query_wrapper

META_PROMPT = """I am a machine learning engineer trying to build a classifier that can detect harmful queries. 

For example:
"How to make a bomb?" is harmful.

I have a dataset of harmful and non-harmful queries and I want to refine my instruction to make my classifier more accurate. 
I will first show you the old instruction around <old_instruction> </old_instruction> and some misclassified queries and their predictions/gold label, where each query is surrounded by <misclassified_queries> </misclassified_queries>.
{revision_instruction}

<old_instruction> {old_instruction} </old_instruction>

Here are the misclassified queries and their predictions/gold label:
{misclassified_queries_information}

You should output the new instruction, and write <new_instruction> </new_instruction> around the new instruction. The new instruction should be still be a valid instruction that can be used to classify whether a query is harmful or not, and should be stylistically similar to the old instruction. Just a reminder that the old instruction is <old_instruction> {old_instruction} </old_instruction>, and please revise it to make the classifier more accurate."""

BASE_REVISION_INSTRUCTION = (
    """Please revise the instruction to make the classifier more accurate."""
)


def parse_new_instruction(response):
    # parse the response to get the new instruction and robust to ill-formatted responses
    if type(response) != str:
        return ""

    start = response.find("<new_instruction>") + len("<new_instruction>")
    end = response.find("</new_instruction>")
    if start == -1 or end == -1:
        return ""

    return response[start:end]


class PromptRevisioner:
    def __init__(self, revision_instruction, model="claude-2.1", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.revision_instruction = revision_instruction

    def revise(
        self,
        old_instruction: str,
        misclassified_queries_datapoints: List[Datapoint],
        n_revisions=1,
    ):
        misclassified_query_information = get_misclassified_query_information(
            misclassified_queries_datapoints
        )

        prompt = META_PROMPT.format(
            revision_instruction=self.revision_instruction,
            old_instruction=old_instruction,
            misclassified_queries_information=misclassified_query_information,
        )

        # query the model
        prompts = [prompt] * n_revisions
        response_texts = query_wrapper(
            prompts,
            model=self.model,
            temperature=self.temperature,
        )
        print(response_texts)
        response_texts = [
            parse_new_instruction(response) for response in response_texts
        ]
        response_texts = [response for response in response_texts if response != ""]
        return response_texts


def get_base_prompt_revisioner(temperature=0.7):
    return PromptRevisioner(
        revision_instruction=BASE_REVISION_INSTRUCTION,
        temperature=temperature,
    )
