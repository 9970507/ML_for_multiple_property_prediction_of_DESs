
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def fp_to_list(mol):
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.tolist()


def col_hash(a):
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=1e308, posinf=1e309, neginf=-1e309)
    return hashlib.md5(a.tobytes()).hexdigest()


def extract_fingerprints(smiles_series, suffix):
    cache = {}
    results = []

    for idx, smi in smiles_series.items():
        if pd.isna(smi):
            results.append([np.nan] * 2048)
            continue

        smi = str(smi).strip()
        if not smi or smi.lower() in {"nan", "none"}:
            results.append([np.nan] * 2048)
            continue

        if smi in cache:
            results.append(cache[smi])
            continue

        try:
            mol = Chem.MolFromSmiles(smi)
            vals = [np.nan] * 2048 if mol is None else fp_to_list(mol)
        except Exception:
            vals = [np.nan] * 2048

        cache[smi] = vals
        results.append(vals)

    columns = [f"MorganFP{i}_{suffix}" for i in range(2048)]
    return pd.DataFrame(results, columns=columns, index=smiles_series.index)


def clean_features(df, min_nonzero=5):
    feat_cols = [
        c for c in df.columns
        if c.startswith("MorganFP") and (c.endswith("_1") or c.endswith("_2"))
    ]

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~X.isna().any(axis=1)
    df = df.loc[valid_mask].reset_index(drop=True)
    X = X.loc[valid_mask].reset_index(drop=True)

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
    df_clean = df.drop(columns=[c for c in df.columns if c in final_drop], errors="ignore")
    return df_clean


def main():
    df = pd.read_csv("Refractive_index.csv")

    df_fp_1 = extract_fingerprints(df["Smiles#1"], "1")
    df_fp_2 = extract_fingerprints(df["Smiles#2"], "2")
    df_full = pd.concat([df, df_fp_1, df_fp_2], axis=1)

    df_clean = clean_features(df_full)
    df_clean.to_csv("Refractive_index_features.csv", index=False)


if __name__ == "__main__":
    main()