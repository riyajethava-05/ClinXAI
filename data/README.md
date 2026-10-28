# Dataset

ClinXAI expects a public diabetes classification CSV at `data/diabetes.csv` with columns:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.

Run `python -m src.train_model` to attempt downloading a public copy. If network access is unavailable, obtain an appropriately licensed copy and place it at that path. Do not use identifiable patient data. This dataset is limited and is for educational demonstration only.
