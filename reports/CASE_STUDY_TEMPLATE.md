# ClinXAI Case Study: Explaining Diabetes Classification with LIME and SHAP

## 1. Abstract
ClinXAI is an educational Explainable AI prototype that demonstrates how a Random Forest classifier can be paired with LIME and SHAP to make model behaviour more interpretable. The model is trained on a public diabetes dataset. It is not intended for clinical use.

## 2. Problem Statement
Machine-learning models can produce predictions that are difficult for people to interpret. In high-impact settings such as healthcare, it is important to examine what features influence a prediction and understand the limits of the model.

## 3. Objectives
- Train and evaluate a binary classification model.
- Explain an individual prediction using LIME.
- Inspect feature contributions using SHAP.
- Compare explanations and discuss limitations.
- Demonstrate responsible-AI considerations such as data quality, bias, privacy and human oversight.

## 4. Dataset and Preprocessing
Dataset: Pima Indians Diabetes Database (confirm the exact source used by the downloaded file).

Features: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function and age. Target: Outcome.

The project validates required columns, converts values to numeric types, removes duplicate rows, splits data into train and test sets, imputes selected zero placeholders using training-set medians, and scales numeric features in a pipeline. These preprocessing choices are assumptions and must be discussed as limitations.

## 5. Model and Evaluation
Model: Random Forest Classifier.

Split: stratified 80% training / 20% test, random seed 42.

Insert the actual values from `reports/evaluation_metrics.json`:
- Accuracy: [fill from actual run]
- Precision: [fill from actual run]
- Recall: [fill from actual run]
- F1-score: [fill from actual run]
- ROC-AUC: [fill from actual run]
- Confusion matrix: [fill from actual run]

Do not claim that these results prove clinical validity.

## 6. LIME Explanation
Run LIME on a selected sample and insert a screenshot of the actual generated chart. Describe the top feature conditions and whether they support or oppose the model's positive-class output. LIME is a local approximation and can vary with sampling and settings.

## 7. SHAP Explanation
Run SHAP on the same sample and insert the actual generated contribution chart. Describe the largest positive and negative SHAP values. SHAP values describe model output contributions; they do not establish causal relationships.

## 8. LIME vs SHAP: Two Observed Examples
### Example 1
- Sample used: [describe a non-identifying demonstration sample]
- LIME top features: [record actual results]
- SHAP top features: [record actual results]
- Agreement/differences: [describe what the generated explanations show]

### Example 2
- Sample used: [describe a second non-identifying demonstration sample]
- LIME top features: [record actual results]
- SHAP top features: [record actual results]
- Agreement/differences: [describe what the generated explanations show]

Do not invent outputs. Generate the explanations in the app and record the results.

## 9. Responsible AI Discussion
The dataset is limited and may not represent all populations. Data collection practices, missing values, label quality, model calibration and subgroup performance can affect reliability. The current prototype does not establish fairness or clinical validity. LIME and SHAP can help inspect model behaviour, but an explanation does not guarantee that the prediction is correct or fair. Real medical use would require appropriate clinical validation, privacy controls, oversight and regulatory review.

## 10. Limitations and Future Scope
- Evaluate on external datasets and independent cohorts.
- Measure subgroup performance and fairness.
- Calibrate probabilities if justified by evaluation.
- Improve dataset documentation and missing-value handling.
- Add clinician-led validation before any real-world use.

## 11. Conclusion
ClinXAI demonstrates an end-to-end explainable classification workflow. It combines model evaluation with local and feature-level explanations and highlights why transparency must be accompanied by careful validation and responsible use.

## 12. References
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?” Explaining the Predictions of Any Classifier.
- Lundberg, S. M., & Lee, S.-I. (2017). “A Unified Approach to Interpreting Model Predictions.”
- scikit-learn: https://scikit-learn.org/
- LIME: https://lime-ml.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
