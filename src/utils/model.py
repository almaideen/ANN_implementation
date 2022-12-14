import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import pandas as pd

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS):

    Layers = [tf.keras.layers.Flatten(input_shape=[28,28],name='inputlayer'),
              tf.keras.layers.Dense(300,activation='relu',name='hiddenlayer1'),
              tf.keras.layers.Dense(100,activation='relu',name='hiddenlayer2'),
              tf.keras.layers.Dense(10,activation='softmax',name='outputlayer')]

    model_clf = tf.keras.models.Sequential(Layers)
    model_clf.summary()
    model_clf.compile(loss = LOSS_FUNCTION, optimizer = OPTIMIZER, metrics = METRICS)

    return model_clf ## untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model,model_name,model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir,unique_filename)
    model.save(path_to_model)


def save_plot(history_data,plot_name,plot_dir):
    pd.DataFrame(history_data.history).plot(figsize=(10, 7), grid=True)
    unique_plotname = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_plotname)
    plt.savefig(path_to_plot)





