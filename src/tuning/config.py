model_tuning = "gru_gan"
path_log = "logs/tuning/" + model_tuning

domain = [
    {'name': 'n_in', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
    # {'name': 'n_out', 'type': 'discrete', 'domain': [1]},
    {'name': 'g_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
    {'name': 'g_dropout', 'type': 'continuous', 'domain': (0, 0.5)},

    {'name': 'd_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
    {'name': 'd_dropout', 'type': 'continuous', 'domain': (0, 0.5)},
    # {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
    # {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
    {'name': 'num_train_d', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
    {'name': 'batch_size', 'type': 'discrete', 'domain': [4, 8, 16, 32]},


]
constraints = []