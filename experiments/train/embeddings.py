import numpy as np
import os

def _load_cached_embeddings(encoder_model_spec: str, dataset_dir: str, dataset_name: str, key_embd_src: str):
    if encoder_model_spec == "OAI":
        encoder_model_spec_str = "oai"
    else:
        encoder_model_spec_str = encoder_model_spec
    key_embds = np.load(
        os.path.join(
            dataset_dir,
            f"{dataset_name}_{encoder_model_spec_str}_embd_{key_embd_src}.npy",
        )
    ).astype("float32")
    if key_embd_src == "answer":
        # If we are using the answer string as the key, we also use it as the value string
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"{dataset_name}_{encoder_model_spec_str}_embd_answer.npy",
            )
        ).astype("float32")
    else:
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"{dataset_name}_{encoder_model_spec_str}_embd_value.npy",
            )
        ).astype("float32")
    return key_embds, value_embds
