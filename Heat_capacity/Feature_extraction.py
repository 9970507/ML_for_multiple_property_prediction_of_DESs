
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def extract_embeddings(smiles_list, tokenizer, model, device, batch_size=64):
    all_embeddings = []
    n = len(smiles_list)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_smiles = smiles_list[start:end]
            encoded = tokenizer(
                batch_smiles, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vec = outputs.pooler_output
            else:
                vec = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(vec.detach().cpu())
    return torch.cat(all_embeddings, dim=0).numpy()


def main():
    model_dir = "MoLFormer-XL-both-10pct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    hidden_size = model.config.hidden_size

    df = pd.read_csv("Heat_capacity.csv")

    all_smiles = pd.concat([df["Smiles#1"], df["Smiles#2"]]).astype(str).unique().tolist()
    embeddings_all = extract_embeddings(all_smiles, tokenizer, model, device)
    smiles_to_vec = {s: embeddings_all[i] for i, s in enumerate(all_smiles)}

    comp1_vectors = np.vstack(df["Smiles#1"].astype(str).map(lambda s: smiles_to_vec[s]).values)
    comp2_vectors = np.vstack(df["Smiles#2"].astype(str).map(lambda s: smiles_to_vec[s]).values)

    for i in range(hidden_size):
        df[f"MoLFormer1_f{i+1}"] = comp1_vectors[:, i]
        df[f"MoLFormer2_f{i+1}"] = comp2_vectors[:, i]

    df.to_csv("Heat_capacity_features.csv", index=False)


if __name__ == "__main__":
    main()
