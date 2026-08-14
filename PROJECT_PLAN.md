# ESBL Prediction Project Plan

## Aim

Develop and evaluate machine-learning models to predict ESBL status in
postoperative *Enterobacterales* infections using information available
before antimicrobial susceptibility testing (AST) results.

## Rationale

Empirical antibiotic treatment is often initiated before AST results are
available. During this period, the organism's susceptibility profile and
resistance phenotype are unknown, potentially resulting in ineffective
treatment or unnecessary broad-spectrum antibiotic use.

## Research question

Can routinely available clinical, microbiological and perioperative
information available before AST results predict ESBL positivity in
postoperative *Enterobacterales* infections?

## Prediction time

**t₀: organism identification, before AST availability.**

Only information available at or before t₀ will be eligible as a
predictor.

## Outcome

Binary ESBL status: 
- ESBL-positive 
- non-ESBL

## Candidate models

Initial models: 
- Logistic regression 
- Random forest 
- XGBoost

| Model                      | Typical behaviour       |
| -------------------------- | ----------------------- |
| Logistic regression        | Linear                  |
| LASSO/elastic-net logistic | Linear + regularisation |
| Random forest              | Nonlinear tree ensemble |
| XGBoost                    | Nonlinear boosted trees |
| RBF-SVM                    | Nonlinear kernel method |


## Model development

An earlier time period will form the development dataset.

Within the development dataset: 
- cross-validation will be grouped by patient; 
- candidate models will undergo automated hyperparameter optimisation using cross-validation; 
- candidate hyperparameter configurations will be compared using mean cross-validated
performance; 
- final hyperparameters will be selected without accessing
the temporal test set; 
- each final model will then be retrained on the
complete development dataset.

The temporal test set will not be used for model or hyperparameter
selection.

## Validation strategy

### Primary: temporal validation

Models will be developed on an earlier period and evaluated once on a
later, completely held-out period.

Provisional split: 
- **Development:** 2015--2022 
- **Temporal test:** 2023--2025

This approximates deployment of a model developed on historical data to
future patients.

### Secondary: random patient-level validation

Performance using a conventional random patient-level split across the
study period will be compared with temporal validation to investigate
whether random splitting produces optimistic performance estimates.

## Data leakage prevention

-   Episodes from the same patient must not occur in both training and
    validation/test partitions.
-   Predictors must be available at or before t₀.
-   No AST-derived or post-prediction information will be included.
-   Future temporal test data must not influence preprocessing, feature
    selection, hyperparameter tuning or model selection.

## Model evaluation

Primary: 
- AUROC 
- AUPRC 
- Calibration

Secondary: 
- Sensitivity 
- Specificity 
- PPV 
- NPV 
- F1 score 
- Brier score

Performance at clinically meaningful probability thresholds will also be
explored.

## Secondary clinical-utility analysis

In a clinically homogeneous subgroup (potentially urinary isolates/UTI
episodes), investigate whether model predictions available before AST
could have supported earlier optimisation of empirical antibiotic
treatment.

This may compare: 
1. treatment prescribed at t₀; 
2. whether that
treatment was ultimately active according to AST; 
3. the treatment decision that would have been supported by the model at a predefined
risk threshold.

## SHAP 

## Next steps

1.  Define the exact development/temporal validation dates.
2.  Audit candidate predictors and confirm availability at t₀.
3.  Define preprocessing and missing-data handling.
4.  Finalise the synthetic-data generator for local development.
5.  Build the preprocessing and model-development pipeline.
6.  Implement grouped cross-validation and hyperparameter optimisation.
7.  Evaluate the locked models on the temporal test set.
8.  Develop the secondary clinical-utility analysis.
9. SHAP 

## Other ideas 

Sensitivity analysis

Repeat the ML comparison using LASSO-selected features only as opposed to all clinically relevant features.


## TO DO 

