import pickle
import numpy as np
import matplotlib.pyplot as plt

from extractor import get_dataframe_by_station_and_pollutant


def load_models(pkl_paths):
    models = []

    for item in pkl_paths:
        for name, path in item.items():
            with open("models/results/"+path, 'rb') as f:
                model = pickle.load(f)
                models.append((name, model))

    return models


def ensemble_predict(models, X):
    predictions = []

    for model in models:
        pred = model.predict(X)
        predictions.append(pred)

    predictions = np.array(predictions)
    ensemble_pred = np.mean(predictions, axis=0)

    return predictions, ensemble_pred

def get_predictions(models):
    predictions = []
    for name, model_data in models:
        pred = model_data['predicted_values']
        predictions.append((name, pred))
    return predictions

def get_ensemble(predictions):

    # weights = [0.2, 0.5]
    return np.mean(
        [pred for _, pred in predictions],
        axis=0
    )


if __name__ == "__main__":

    pkl_files = [
        {"SVR": "1-svr/1-svr(tw12)-020426125514.pkl"},
    ]

    ensemble = True

    pollutant = "MP10"
    station_code = "SP71"
    #Entry
    df = get_dataframe_by_station_and_pollutant(station_code, pollutant)

    ts = df['actual']
    
    models = load_models(pkl_files)
    y_test = models[0][1]['real_values']
    X_test = np.arange(1, 21).reshape(-1, 1)

    predictions = get_predictions(models=models)

    ensemble_pred = get_ensemble(predictions=predictions)

    test_index = ts.index[-len(ensemble_pred):]

    # PLOT
    plt.figure(figsize=(12, 6))

    plt.plot(test_index, y_test, label="Real", linewidth=3)

    #Predictions
    for name, pred in predictions:
        plt.plot(test_index, pred, label=name, alpha=0.7)
    
    # 🔥 Detach do ensemble
    plt.plot(test_index, ensemble_pred, linestyle="--", linewidth=3, label="Ensemble")

    plt.title("Models Comparison")
    plt.xlabel("Time Step - D")
    plt.ylabel(pollutant)

    plt.legend()
    plt.grid()

    plt.show()