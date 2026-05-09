# Machine learning for predicting physicochemical properties of deep eutectic solvents

## Overview

This repository accompanies a study that establishes a systematic XGBoost-based machine learning framework for predicting six key physicochemical properties of Type III/V deep eutectic solvents (DESs): melting point, viscosity, density, heat capacity, electrical conductivity, and surface tension. In the paper, multiple molecular representations — explicit descriptors, Morgan fingerprints, ChemBERTa embeddings, MoLFormer embeddings, and their hybrid combinations — were systematically compared to identify the optimal feature strategy for each property. This repository provides the source code and data corresponding to the optimal representation identified for each property, along with SHAP-based interpretability analysis.

## Repository Structure

```
ML_for_multiple_property_prediction_of_DESs/
├── Melting_point/
├── Density/
├── Viscosity/
│   └── ChemBERTa-77M-MLM/        # Pretrained ChemBERTa model
├── Heat_capacity/
├── Electrical_conductivity/
├── Surface_tension/
├── LICENSE
├── README.md
└── requirements.txt
```

## Code Modules

**Feature_extraction.py** extracts molecular features from SMILES strings. The representation method varies by property: explicit RDKit descriptors for melting point, density, surface tension, and electrical conductivity; hybrid ChemBERTa embeddings combined with RDKit descriptors for viscosity; and MoLFormer embeddings for heat capacity. For RDKit-based properties, uninformative features are removed after extraction.

**Model_training.py** performs mixture-based train/test splitting via `GroupShuffleSplit`, ensuring no DES system appears in both sets. XGBoost is trained with `RandomizedSearchCV` under `GroupKFold` cross-validation, and evaluated on an independent test set using R² and RMSE.

**SHAP_analysis.py** computes SHAP values using `TreeExplainer` on the trained XGBoost model and generates beeswarm plots for feature importance visualization.

*Note: All scripts provided here are streamlined versions intended to demonstrate the core methodology. Full implementations including additional analyses are available from the corresponding author upon request.*

## Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Key Libraries

- **pandas** and **numpy**: Data manipulation and numerical computation.
- **scikit-learn**: Data splitting, cross-validation, and hyperparameter optimization.
- **xgboost**: Gradient boosting model for property prediction.
- **rdkit**: Molecular descriptor extraction from SMILES.
- **transformers** and **torch**: Pretrained molecular language model (ChemBERTa / MoLFormer) embedding extraction.
- **shap**: Model interpretability and feature importance analysis.
- **joblib**: Model serialization.
- **matplotlib**: Visualization.
