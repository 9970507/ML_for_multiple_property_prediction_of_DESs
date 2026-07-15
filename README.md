# Machine learning for predicting multiple physicochemical properties of Type III/V deep eutectic solvents

## Overview

This repository accompanies the study “Machine learning for predicting multiple physicochemical properties of Type III/V deep eutectic solvents using low-cost and interpretable molecular representations.” The study develops an XGBoost framework for predicting six physicochemical properties: melting point, viscosity, density, heat capacity, electrical conductivity, and refractive index. Four SMILES-derived molecular representations, including explicit descriptors, Morgan fingerprints, ChemBERTa embeddings, and MoLFormer embeddings, together with their hybrids, were compared under a strict “mixtures out” evaluation protocol. The final property-specific models use explicit descriptors for density, heat capacity, and electrical conductivity, and Morgan fingerprints for melting point, refractive index, and viscosity. This repository provides the datasets, source code, trained models, and SHAP analysis scripts corresponding to the final models.

## Repository Structure

```
ML_for_multiple_property_prediction_of_DESs/
├── Melting_point/
├── Density/
├── Viscosity/
├── Heat_capacity/
├── Electrical_conductivity/
├── Refractive_index/
├── LICENSE
├── README.md
└── requirements.txt
```

Each property folder contains the corresponding dataset, training and test sets, feature-extraction script, dataset-splitting script, model-training script, SHAP analysis script, and serialized final model.

## Code Modules

**Feature_extraction.py**

Generates molecular features from the SMILES strings of the DES components. Morgan fingerprints with a radius of 2 and 2,048 bits are generated for melting point, viscosity, and refractive index, whereas RDKit molecular descriptors are calculated for density, heat capacity, and electrical conductivity. Constant, duplicate, and sparse features are removed before model development.

**Dataset_Splitting.py**

Splits each feature dataset into training and test sets under the "mixtures out" protocol. Samples are grouped by their complete `Smiles#1` and `Smiles#2` combination so that all measurements for a given DES mixture are assigned entirely to either the training set or the test set. The scripts use a fixed random seed of 42 and target an approximately 80:20 split.

**Model_training.py**

Trains and evaluates the property-specific XGBoost models using the supplied `Train_set.csv` and `Test_set.csv` files. Hyperparameters are optimized using randomized search with five-fold grouped cross-validation. Model performance is evaluated using the coefficient of determination (R²) and root mean squared error (RMSE).

**SHAP_analysis.py**

Loads the trained model and calculates SHAP values using `TreeExplainer` to identify the features associated with model predictions. The resulting feature contributions are visualized using SHAP beeswarm plots.

*Note: All scripts provided here are streamlined versions intended to demonstrate the core methodology. Full implementations including additional analyses are available from the corresponding author upon request.*

## Dependencies

The code was developed using Python 3.9.21. The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Key Libraries

- **pandas** and **numpy**: Data manipulation and numerical computation.
- **scikit-learn**: Data splitting, cross-validation, and hyperparameter optimization.
- **xgboost**: Gradient boosting model for property prediction.
- **rdkit**: Molecular descriptor extraction from SMILES.
- **shap**: Model interpretability and feature importance analysis.
- **joblib**: Model serialization.
- **matplotlib**: Visualization.
