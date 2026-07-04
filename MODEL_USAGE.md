# Model Usage

## Train the Model

```bash
python scripts/build_autism_screening_analysis.py
```

This command regenerates:

- Cleaned dataset
- EDA charts
- Model comparison metrics
- Confusion matrix
- Feature importance
- Sample predictions
- Saved model artifact
- Arabic executive report

## Run a Sample Prediction

```bash
python scripts/predict_autism_screening.py
```

The trained model is saved at:

```text
models/best_autism_screening_model.joblib
```

The model expects these input fields:

```text
a1_score, a2_score, a3_score, a4_score, a5_score,
a6_score, a7_score, a8_score, a9_score, a10_score,
age, gender, ethnicity, jaundice, family_asd,
country_of_residence, used_app_before, relation
```

## Medical Disclaimer

The output is a machine learning screening estimate only. It is not a medical diagnosis, clinical decision tool, or substitute for assessment by qualified professionals.
