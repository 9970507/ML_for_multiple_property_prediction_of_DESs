
import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("Density_features.csv")

    mixture_labels = (
        df["Smiles#1"].astype(str).str.strip()
        + "||"
        + df["Smiles#2"].astype(str).str.strip()
    )

    mixture_counts = mixture_labels.value_counts()
    mixtures = mixture_counts.index.tolist()

    rng = np.random.RandomState(42)
    rng.shuffle(mixtures)

    target_test_n = int(round(len(df) * 0.2))
    test_mixtures = set()
    test_count = 0

    for mixture in mixtures:
        mixture_size = mixture_counts[mixture]
        if test_count + mixture_size <= target_test_n * 1.1:
            test_mixtures.add(mixture)
            test_count += mixture_size
        if test_count >= target_test_n:
            break

    test_mask = mixture_labels.isin(test_mixtures)

    train_set = df.loc[~test_mask].reset_index(drop=True)
    test_set = df.loc[test_mask].reset_index(drop=True)

    train_set.to_csv("Train_set.csv", index=False)
    test_set.to_csv("Test_set.csv", index=False)


if __name__ == "__main__":
    main()