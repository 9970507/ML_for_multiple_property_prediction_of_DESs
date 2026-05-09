
import hashlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def get_all_descriptors_list():
    return [x[0] for x in Descriptors.descList]


def sanitize_values(values):
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr.tolist()


def col_hash(a):
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=1e308, posinf=1e309, neginf=-1e309)
    return hashlib.md5(a.tobytes()).hexdigest()


def extract_chemberta_embeddings(smiles_list, tokenizer, model, device, batch_size=64):
    all_embeddings = []
    n = len(smiles_list)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_smiles = smiles_list[start:end]
            encoded = tokenizer(
                batch_smiles, padding=True, truncation=True, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            vec = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(vec.detach().cpu())
    return torch.cat(all_embeddings, dim=0).numpy()


def extract_rdkit_descriptors(smiles_series, suffix, desc_names, calc):
    cache = {}
    results = []
    for idx, smi in smiles_series.items():
        if pd.isna(smi):
            results.append([np.nan] * len(desc_names))
            continue
        smi = str(smi).strip()
        if not smi or smi.lower() in {"nan", "none"}:
            results.append([np.nan] * len(desc_names))
            continue
        if smi in cache:
            results.append(cache[smi])
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                vals = [np.nan] * len(desc_names)
            else:
                vals = sanitize_values(calc.CalcDescriptors(mol))
            cache[smi] = vals
            results.append(vals)
        except Exception:
            vals = [np.nan] * len(desc_names)
            cache[smi] = vals
            results.append(vals)
    new_cols = [f"{name}_{suffix}" for name in desc_names]
    return pd.DataFrame(results, columns=new_cols, index=smiles_series.index)


def clean_rdkit_features(df, min_nonzero=10):
    feat_cols = [c for c in df.columns if c.endswith("_1") or c.endswith("_2")]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)

    nunique = X.nunique(dropna=False)
    drop_constant = sorted(X.columns[(nunique <= 1)].tolist())
    X1 = X.drop(columns=drop_constant)

    hashes = {}
    drop_dup = []
    for c in X1.columns:
        h = col_hash(X1[c].values)
        if h in hashes:
            drop_dup.append(c)
        else:
            hashes[h] = c
    drop_dup = sorted(set(drop_dup))
    X2 = X1.drop(columns=drop_dup)

    drop_rare = sorted(X2.columns[((X2 != 0).sum() < min_nonzero)].tolist())

    final_drop = set(drop_constant) | set(drop_dup) | set(drop_rare)
    remaining = [c for c in feat_cols if c not in final_drop]
    return remaining


def main():
    model_dir = "ChemBERTa-77M-MLM"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    chemberta_model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    chemberta_model.to(device)
    chemberta_model.eval()
    hidden_size = chemberta_model.config.hidden_size

    df = pd.read_csv("Viscosity.csv")

    all_smiles = pd.concat([df["Smiles#1"], df["Smiles#2"]]).astype(str).unique().tolist()
    embeddings_all = extract_chemberta_embeddings(all_smiles, tokenizer, chemberta_model, device)
    smiles_to_vec = {s: embeddings_all[i] for i, s in enumerate(all_smiles)}

    comp1_vectors = np.vstack(df["Smiles#1"].astype(str).map(lambda s: smiles_to_vec[s]).values)
    comp2_vectors = np.vstack(df["Smiles#2"].astype(str).map(lambda s: smiles_to_vec[s]).values)

    for i in range(hidden_size):
        df[f"ChemBERTa1_f{i+1}"] = comp1_vectors[:, i]
        df[f"ChemBERTa2_f{i+1}"] = comp2_vectors[:, i]

    desc_names = get_all_descriptors_list()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    df_desc_1 = extract_rdkit_descriptors(df["Smiles#1"], "1", desc_names, calc)
    df_desc_2 = extract_rdkit_descriptors(df["Smiles#2"], "2", desc_names, calc)
    df_with_desc = pd.concat([df, df_desc_1, df_desc_2], axis=1)

    clean_desc_cols = clean_rdkit_features(df_with_desc)

    chemberta_cols = [c for c in df_with_desc.columns if c.startswith("ChemBERTa1_") or c.startswith("ChemBERTa2_")]
    non_feature_cols = [c for c in df.columns if not c.startswith("ChemBERTa")]
    keep_cols = non_feature_cols + chemberta_cols + clean_desc_cols

    df_final = df_with_desc[keep_cols]
    df_final.to_csv("Viscosity_features.csv", index=False)


if __name__ == "__main__":
    main()
