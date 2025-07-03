import torch
import numpy as np
from typing import Callable, Dict, List
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa, get_i_dont_know_ans

def get_batch(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable, torch.Tensor], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    B: int = 20,
    random_sample=True,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
):
    """
    dataset: List of dictionary, denoting the KB, used to extract QA pairs
    model: The LLM, used to provide the embedding
    kb_embedding: KB embedding (differentiable)
    B: Batchsize
    include_outlier : Create a batch of question without answer in the KB.
    multi_entities : Create a batch of question that involves more than one entities.
    """
    labels = []
    if multi_entities is not None:
        assert not include_outlier

    if random_sample:
        if multi_entities is not None:
            batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)
        else:
            batch_indices = np.random.choice(len(dataset), B, replace=False)
    else:
        batch_indices = np.arange(B)

    def get_question_and_answer(idx: int) -> tuple[str, str]:
        if use_extended_qa:
            Q, A = dataset[idx]["extended_Q"], dataset[idx]["extended_A"]

        elif multi_entities is not None:
            Q, A = generate_multi_entity_qa(
                [dataset[i]["name"] for i in idx],
                [dataset[i]["description_type"] for i in idx],
                [dataset[i]["description"] for i in idx],
            )
        else:
            Q = augment_row(dataset[idx]) if use_data_aug else dataset[idx]["Q"]
            A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]
        return Q, A

    with torch.autograd.no_grad():
        input_strs = []
        real_batch_indices = []
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            if Q is not None and A is not None:
                input_strs.append(qa_format_func(Q, A))
                real_batch_indices.append(idx)
            else:
                print("Q or Answer is none")
        batch_indices = real_batch_indices
        tokenizer_output = tokenizer(
            input_strs, return_tensors="pt", padding=True, return_offsets_mapping=True
        ).to(device)
        input_ids, attention_masks, offset_mapping = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
            tokenizer_output["offset_mapping"],
        )

        labels = label_func(input_ids, input_strs, tokenizer, offset_mapping)
    if include_outlier:
        # Generate a new set of indices, such that the KB does not contain the entity where the question comes from
        batch_indices = np.random.choice(len(dataset), B, replace=False)
    return input_ids, attention_masks, labels, batch_indices
