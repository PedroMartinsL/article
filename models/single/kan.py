import os
os.environ["PYTORCH_LIGHTNING_VERBOSITY"] = "0"

from services.fit_predict import FitPrediction
from neuralforecast.models import KAN

FitPrediction.execute(
    model_execs=10,
    data_title='kan',
    model=KAN,
    parameters={
        'input_size': [24],
        'hidden_size': [32, 64],
        'learning_rate': [1e-3],
        'max_steps': [100],
        'scaler_type': ['standard'],
    },
    normalize="min_max_scaler",
    differencing=True
)