from services.fit_predict import FitPrediction
from neuralforecast.models import NHITS

FitPrediction.execute(
    model_execs=5,
    data_title='nhits',
    model=NHITS,
    parameters={
        'input_size': [24, 96],
        'n_freq_downsample': [[168, 24, 1], [24, 12, 1]],
        'mlp_units': [[[512, 512], [512, 512], [512, 512]]],
        'learning_rate': [1e-3],
        'max_steps': [300],
        'scaler_type': ['standard'],
    },
    differencing=True
)