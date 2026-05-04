from services.fit_predict import FitPrediction
from neuralforecast.models import NHITS

FitPrediction.execute(
    model_execs=10,
    data_title='nhits',
    model=NHITS,
    parameters={
        'input_size': [12, 24],
        'learning_rate': [1e-3],
        'max_steps': [100],
        'batch_size': [32],
        'scaler_type': ['robust'],
    },
    normalize="min_max_scaler",
    differencing=True
)