from typing import Dict, List

from tqdm import tqdm

from bigcode_eval.base import Task

from collections import defaultdict

_CITATION = """
@article{allal2023santacoder,
  title={SantaCoder: don't reach for the stars!},
  author={Allal, Loubna Ben and Li, Raymond and Kocetkov, Denis and Mou, Chenghao and Akiki, Christopher and Ferrandis, Carlos Munoz and Muennighoff, Niklas and Mishra, Mayank and Gu, Alex and Dey, Manan and others},
  journal={arXiv preprint arXiv:2301.03988},
  year={2023}
}
"""

LANGUAGES = [
    "py",
    "js",
    "java",
]


def create_all_tasks():
    return {
        "santacoder_fim": SantaCoderFIM,
        "starcoder_fim": StarCoderFIM,
    }


def initialize_empty_metrics(languages: List[str]) -> Dict[str, float]:
    metrics = {}
    for lang in languages:
        metrics[f"n_accurate_{lang}"] = 0.0
        metrics[f"n_count_{lang}"] = 0.0
    return metrics


def aggregate_per_lang_accuracy(
    metrics: Dict[str, float], languages: List[str]
) -> Dict[str, float]:
    em_metrics = {}
    for lang in languages:
        # avoid div by 0
        acc = (
            metrics[f"n_accurate_{lang}"] / metrics[f"n_count_{lang}"]
            if metrics[f"n_count_{lang}"]
            else 0
        )
        em_metrics[f"{lang} Exact Match"] = acc

    return em_metrics


class SantaCoderFIM(Task):
    DATASET_PATH = "bigcode/santacoder-fim-task"

    def __init__(
        self,
        fim_prefix: str = "<fim-prefix>",
        fim_middle: str = "<fim-middle>",
        fim_suffix: str = "<fim-suffix>",
        stop_words: List[str] = ["<|endoftext|>", "<|filename|>"],
        requires_execution: bool = False
    ):
        super().__init__(
            stop_words=stop_words,
            requires_execution=requires_execution,
        )
        self.fim_prefix = fim_prefix
        self.fim_middle = fim_middle
        self.fim_suffix = fim_suffix

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["train"]
        
        # Filter only Python examples
        py_dataset = [doc for doc in dataset if doc["language"] == "py"]
        return py_dataset
        # return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return f"""{self.fim_prefix}{doc["prompt"]}{self.fim_suffix}{doc["suffix"]}{self.fim_middle}"""

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["canonical_solution"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        output = generation[len(prompt) :]
        return self._stop_at_stop_token(output, self.stop_words)
        # return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        
        Evaluates generations against references and returns:
        - aggregated metrics per language
        - per-sample exact match info for optional deeper analysis
        """
        metrics = initialize_empty_metrics(LANGUAGES)
        exact_match_results = defaultdict(list)  # {task_id: [ (completion_id, result_dict), ... ]}
        for idx, (gen, reference) in tqdm(enumerate(zip(generations, references))):
            language = self.get_dataset()[idx]["language"]
            # for g in gen:
            #     metrics[f"n_accurate_{language}"] += int(g.strip() == reference.strip())
            for completion_id, g in enumerate(gen):
                exact = int(g.strip() == reference.strip())
                metrics[f"n_accurate_{language}"] += exact

                result_info = {
                    "task_id": idx,
                    "completion_id": completion_id,
                    "exact_match": exact
                }

                exact_match_results[idx].append((completion_id, result_info))


            metrics[f"n_count_{language}"] += len(gen)

        em_metrics = aggregate_per_lang_accuracy(metrics, LANGUAGES)

        return em_metrics, exact_match_results

    # return em_metrics


class StarCoderFIM(SantaCoderFIM):
    DATASET_PATH = "bigcode/santacoder-fim-task"

    def __init__(self):
        fim_prefix = "<fim_prefix>"
        fim_middle = "<fim_middle>"
        fim_suffix = "<fim_suffix>"
        stop_words = ["<|endoftext|>", "<|filename|>", "<file_sep>"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=False,
            fim_prefix=fim_prefix,
            fim_middle=fim_middle,
            fim_suffix=fim_suffix,
        )
