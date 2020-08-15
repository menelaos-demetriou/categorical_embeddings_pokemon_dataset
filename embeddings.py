import os

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, Embedding, Flatten

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def process_data(pokemon):

    num_attribs = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

    # Create preprocess transformations
    preprocess = ColumnTransformer([("num", StandardScaler(), num_attribs)])
    data_transformed = preprocess.fit_transform(pokemon.copy())

    # Convert True/False to binary
    pokemon["Legendary"] = pokemon["Legendary"].astype(int)

    columns = ["Type 1", "Type 2", "Gen"] + num_attribs + ["Legendary"]
    df = pd.DataFrame(columns=columns)
    df["Type 1"] = pokemon["Type 1"]
    df["Type 2"] = pokemon["Type 2"]
    df["Gen"] = pokemon["Generation"]
    df[num_attribs] = data_transformed
    df["Legendary"] = pokemon["Legendary"]

    return df


def create_model(metrics, embedding_size):
    # create model
    input_type = keras.layers.Input(shape=(2,))
    input_gen = keras.layers.Input(shape=(1,))
    input_stats = keras.layers.Input(shape=(7,))

    model = Sequential()
    model.add(Embedding(input_dim=19, output_dim=embedding_size, input_length=2, name="embedding-layer"))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=metrics)
    return model


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.8, 1])
        else:
          plt.ylim([0, 1])

        plt.legend()


def main():
    # Read csv file
    pokemon = pd.read_csv("data/pokemon.csv")
    pokemon["Type 2"] = pokemon["Type 2"].fillna("no_type")

    processed = process_data(pokemon)

    x = pokemon[["Type 1", "Type 2"]].copy()
    y = pokemon["Legendary"].copy()

    # Split dataset to train, val and test set
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    embedding_size = 3
    epochs = 50
    batch_size = 2048

    # Get metrics
    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # Get early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    # Define checkpoints

    # Create model
    model = create_model(metrics, embedding_size)
    # Get summary
    model.summary()

    # model.fit(x=data_small_df['mnth'].as_matrix(), y=data_small_df['cnt_Scaled'].as_matrix(), epochs=50, batch_size=4)
    # Check first column


if __name__ == "__main__":
    main()
