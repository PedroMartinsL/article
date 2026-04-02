# # model disponible at https://github.com/albertbup/deep-belief-network/blob/master/README.md

# import warnings
# from services.fit_predict import FitPrediction

# from deep_belief_network.dbn.models import SupervisedDBNRegression

# model_execs = 10
# data_title = 'mlp'


# parameters = {'hidden_layers_structure': [100, 200], 
#                   'learning_rate_rbm': [0.01,0.001],
#                   'learning_rate': [0.01,0.001],
#                   'time_window': [12]
#                  }
# model = SupervisedDBNRegression(n_epochs_rbm=20,
#                                 n_iter_backprop=200,
#                                 batch_size=16,
#                                 activation_function='relu',  verbose=False)

# FitPrediction.train_sklearn(model_execs, data_title, parameters, model)