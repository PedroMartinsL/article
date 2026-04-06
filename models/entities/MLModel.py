import os
import pickle
from matplotlib import pyplot as plt
import pandas as pd

class MLModel():
    def __init__(self, name: str, **model_data: dict):
        """
        MLModel(
            name="MyModel",
            test_metrics=...,
            val_metrics=...,
            train_metrics=...,
            real_values=...,
            predicted_values=...,
            pool_prevs=...,
            params=...
        )
        """
        self.name = name
        self.test_metrics = model_data.get('test_metrics', None)
        self.val_metrics = model_data.get('val_metrics', None)
        self.train_metrics = model_data.get('train_metrics', None)
        self.real_values = model_data.get('real_values', None)
        self.predicted_values = model_data.get('predicted_values', None)
        self.pool_prevs = model_data.get('pool_prevs', None)
        self.params = model_data.get('params', None)

    def get_predictions(models: list):
        predictions = []
        for model_cls in models:
            name = model_cls.name
            pred = model_cls.predicted_values
            predictions.append((name, pred))
        return predictions
    
    def load_models(pkl_paths: dict):
        """
            Load models by path

            e.g.

            pkl_files = [
                {"ARIMA": "1-arima/1-arima-050426221319.pkl"},
            ]
        """
        models = []

        for item in pkl_paths:
            for name, path in item.items():
                with open(os.path.join("models", "results", path), 'rb') as f:
                    model_data = pickle.load(f)
                    class_model = MLModel(name, **model_data)
                    models.append(class_model)

        return models
    
    def plot_test_metrics_table(models):
        """
            models: list of MLModel or EnsembleModel objects
            Each object must have the attribute .test_metrics (a dictionary of metrics)
        """
    
        data = []
        for m in models:
            row = {'Model': m.name}
            row.update(m.test_metrics)
            data.append(row)

        df = pd.DataFrame(data)
        
        # PLotting table
        fig, ax = plt.subplots(figsize=(10, len(models)*0.5 + 1))
        ax.axis('off')
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        plt.title("Models Test Metrics", fontsize=14)
        plt.show()

    # def plot_perfomance(test_index, y_test, ensemble_pred, y_label, predictions):
    #     # PLOT
    #     plt.figure(figsize=(12, 6))

    #     plt.plot(test_index, y_test, label="Real", linewidth=3)

    #     #Predictions
    #     for name, pred in predictions:
    #         plt.plot(test_index, pred, label=name, alpha=0.7)
        
    #     # Detach do ensemble
    #     plt.plot(test_index, ensemble_pred, linestyle="--", linewidth=3, label="Ensemble")

    #     plt.title("Models Comparison")
    #     plt.xlabel("Time Step - D")
    #     plt.ylabel(y_label)

    #     plt.legend()
    #     plt.grid()

    #     plt.show()

    def plot_perfomance(test_index, y_test, ensemble_pred, y_label, predictions):
        import matplotlib.pyplot as plt
        import numpy as np

        # Garantir arrays 1D
        y_test = np.asarray(y_test).reshape(-1)
        ensemble_pred = np.asarray(ensemble_pred).reshape(-1)
        test_index = np.asarray(test_index)

        # Descobrir menor tamanho global
        lengths = [len(test_index), len(y_test), len(ensemble_pred)]
        lengths += [len(np.asarray(pred).reshape(-1)) for _, pred in predictions]

        min_len = min(lengths)

        # Alinhar tudo
        test_index = test_index[-min_len:]
        y_test = y_test[-min_len:]
        ensemble_pred = ensemble_pred[-min_len:]

        plt.figure(figsize=(12, 6))

        # Real
        plt.plot(test_index, y_test, label="Real", linewidth=3)

        # Predictions individuais
        for name, pred in predictions:
            pred = np.asarray(pred).reshape(-1)
            pred = pred[-min_len:]
            plt.plot(test_index, pred, label=name, alpha=0.7)

        # Ensemble
        plt.plot(test_index, ensemble_pred, linestyle="--", linewidth=3, label="Ensemble")

        plt.title("Models Comparison")
        plt.xlabel("Time Step - D")
        plt.ylabel(y_label)

        plt.legend()
        plt.grid()

        plt.show()