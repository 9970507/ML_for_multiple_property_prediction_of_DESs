
import hashlib
import numpy as np
import pandas as pd
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


def extract_descriptors(smiles_series, suffix, desc_names, calc):
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


def clean_features(df, min_nonzero=10):
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
    df_clean = df.drop(columns=[c for c in df.columns if c in final_drop], errors="ignore")
    return df_clean


def main():
    df = pd.read_csv("Surface_tension.csv")

    desc_names = get_all_descriptors_list()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    df_desc_1 = extract_descriptors(df["Smiles#1"], "1", desc_names, calc)
    df_desc_2 = extract_descriptors(df["Smiles#2"], "2", desc_names, calc)
    df_full = pd.concat([df, df_desc_1, df_desc_2], axis=1)

    df_clean = clean_features(df_full)
    df_clean.to_csv("Surface_tension_features.csv", index=False)


if __name__ == "__main__":
    main()
