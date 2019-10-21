from data import DataSets
import numpy as np
import pandas as pd
import utils
import model_zoo
from metrics import evaluate


def run(model, config, dataset: DataSets):
    x_train, x_test, y_train, y_test = dataset.get_data(0.2)

    model = getattr(model_zoo, model)(**config['model'])
    model.fit(x_train, y_train, **config['train'])

    pred = model.predict(x_test)
    model.close_session()

    pred = np.reshape(pred, (-1, y_test.shape[-1]))
    y_test = np.reshape(y_test, (-1, y_test.shape[-1]))

    pred_invert = dataset.invert_transform(pred)
    y_test_invert = dataset.invert_transform(y_test)

    result_eval = evaluate(y_test_invert, pred_invert, metrics=['mae', 'rmse', 'mape', 'smape'])

    print("Result test:", result_eval)

    short = ['actual', 'predict']
    long = ['actual_cpu', 'actual_mem', 'predict_cpu', 'predict_mem']
    columns = short

    a = np.concatenate([y_test, pred], axis=1)
    df = pd.DataFrame(a, columns=columns)
    utils.plot_distribution([df[col].values for col in columns], legends=columns)
    utils.plot_time_series(df)

    b = np.concatenate([y_test_invert, pred_invert], axis=1)
    df_2 = pd.DataFrame(b, columns=columns)
    utils.plot_distribution([df_2[col].values for col in columns], legends=columns)
    utils.plot_time_series(df_2, title=f"{result_eval.values()}")
    utils.plt.show()
