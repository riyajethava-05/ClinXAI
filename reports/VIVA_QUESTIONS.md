# ClinXAI Viva Questions and Answers

**1. What is the objective of ClinXAI?**  
To demonstrate a diabetes classification model and explain its predictions using LIME and SHAP.

**2. What is Explainable AI?**  
Explainable AI includes methods that help people understand how a model arrives at its outputs.

**3. Why use Random Forest?**  
It combines predictions from multiple decision trees and can model non-linear relationships in tabular data.

**4. What is classification?**  
Classification predicts a discrete class label, such as the dataset's Outcome label 0 or 1.

**5. What is LIME?**  
LIME approximates a model locally around one input using a simpler interpretable model.

**6. What is SHAP?**  
SHAP assigns feature contribution values based on Shapley-value ideas to help explain model outputs.

**7. What is the difference between local and global explanation?**  
A local explanation focuses on one prediction. A global analysis summarizes patterns across many predictions.

**8. Why do LIME and SHAP sometimes disagree?**  
They use different mathematical approaches, sampling assumptions and definitions of feature contribution.

**9. What is data leakage?**  
Data leakage occurs when information from outside the training process improperly influences model fitting or evaluation.

**10. Why use a train/test split?**  
To evaluate the model on data that was not used to fit it.

**11. What is precision?**  
Precision is the fraction of predicted positive cases that are actually positive in the labelled test data.

**12. What is recall?**  
Recall is the fraction of actual positive cases in the test data that the model correctly identifies.

**13. What is the F1-score?**  
The harmonic mean of precision and recall.

**14. What is ROC-AUC?**  
A measure of how well model scores rank positive examples above negative examples across thresholds.

**15. Do SHAP values prove causation?**  
No. They describe how features contribute to the model's output, not real-world causal effects.

**16. Does a high model probability mean a person has that exact medical risk?**  
No. It is a model output and may not be calibrated; this educational model is not a diagnosis.

**17. What are responsible-AI concerns in this project?**  
Dataset representativeness, bias, privacy, uncertainty, explainability limits and human oversight.

**18. Why use Streamlit?**  
It makes it possible to build an interactive Python web app without a separate frontend framework.

**19. Why impute missing values?**  
Many models cannot directly handle missing values. Imputation fills missing entries using a defined rule.

**20. What are future improvements?**  
External validation, fairness testing, probability calibration, better dataset documentation and expert review.
