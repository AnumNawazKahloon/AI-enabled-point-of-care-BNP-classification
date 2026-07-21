# BNP CNN Ensemble Classifier

5-model 2D-CNN ensemble (+ stacking meta-learner) for multi-class BNP concentration
classification from EIS (impedance) features.

## Data split

Splitting is **patient-based only** — no random row-level split. All rows belonging
to the same patient are kept together in a single split (train, test, or val), so
no patient's data ever leaks across splits. Patients are assigned to train/test/val
(70/20/10) per class, based on each patient's majority class label.

## Data layout

Place these files under `users/tang-lab/Anum/BNP/`:

| File | Description |
|---|---|
| `patient_data_5time.csv` | Feature rows (headerless): 720 features + last column = class label |
| `patient_ids.csv` | One patient ID per row, same row order as `patient_data_5time.csv` |
| `patient_data.csv` | *(optional)* external hold-out test set, same format as above |

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, hyperparameters, constants |
| `data_utils.py` | CSV loading, patient-based split, scaling, augmentation |
| `models.py` | 5 CNN architectures (shallow, deep, wide, residual, inception) |
| `ensemble.py` | Soft/weighted/hard/temperature voting, stacking, `CNNEnsemble` production wrapper |
| `evaluate.py` | Confusion matrix, ROC, calibration, diversity, BNP correlation, Nyquist plots |
| `train.py` | Main entry point — runs the full pipeline |

## Usage

```bash
pip install -r requirements.txt
python train.py
```

Outputs (trained models, scaler, meta-learner, figures) are written to `outputs/`.

## Production inference

```python
import joblib
from ensemble import CNNEnsemble
from tensorflow import keras

models = {n: keras.models.load_model(f'outputs/ensemble_models/{n}.keras')
          for n in ['CNN_A', 'CNN_B', 'CNN_C', 'CNN_D', 'CNN_E']}
scaler = joblib.load('outputs/ensemble_models/ensemble_scaler.pkl')
meta_lr = joblib.load('outputs/ensemble_models/meta_learner.pkl')

ensemble = CNNEnsemble(models, scaler, meta_lr, method='weighted')
preds, probas = ensemble.predict(X_new)  # X_new: (N, 720) raw features
```
