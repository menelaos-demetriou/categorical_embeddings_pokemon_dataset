import os

import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras.models import Model
from tensorboard.plugins import projector
from keras.utils.vis_utils import plot_model
from ann_visualizer.visualize import ann_viz
from keras.layers import Dense, Embedding, Flatten, concatenate, Input

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
NUM_ATTRIB = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]


def process_data(pokemon):
    # Create preprocess transformations
    preprocess = ColumnTransformer([("num", StandardScaler(), NUM_ATTRIB), ])
    data_transformed = preprocess.fit_transform(pokemon.copy())

    # Convert True/False to binary
    pokemon["Legendary"] = pokemon["Legendary"].astype(int)

    columns = ["combined_type", "Gen"] + NUM_ATTRIB + ["Legendary"]
    df = pd.DataFrame(columns=columns)
    pokemon["combined_type"] = pokemon["Type 1"] + "-" + pokemon["Type 2"]
    types = pokemon["combined_type"].unique()
    type_dict = dict(zip(types, range(len(types))))
    pokemon = pokemon.replace(type_dict)

    df["combined_type"] = pokemon["combined_type"]
    pokemon['Generation'] = pokemon['Generation'] - 1
    pokemon['Generation'] = pokemon['Generation'].astype(int)
    df["Gen"] = pokemon["Generation"]
    df[NUM_ATTRIB] = data_transformed
    df["Legendary"] = pokemon["Legendary"].astype("int32")

    return df, type_dict


def create_model(metrics, embedding_size):
    # create model
    input_type = Input(shape=(1,))
    input_gen = Input(shape=(1,))
    input_stats = Input(shape=(7,))

    embedding_type = Embedding(input_dim=154, output_dim=embedding_size, input_length=1,
                               name="embedding-type")(input_type)
    embedding_gen = Embedding(input_dim=6, output_dim=embedding_size, input_length=1, name="embedding-gen")(input_gen)

    flatt_type = Flatten()(embedding_type)
    flatt_gen = Flatten()(embedding_gen)

    hidden_type = Dense(50, activation="relu")(flatt_type)
    hidden_gen = Dense(50, activation="relu")(flatt_gen)
    hidden_stats = Dense(50, activation="relu")(input_stats)

    merged = concatenate([hidden_type, hidden_gen, hidden_stats])

    hidden = Dense(15, activation="relu")(merged)

    output = Dense(1, activation="sigmoid")(hidden)

    model = Model(inputs=[input_type, input_gen, input_stats], outputs=output)

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=metrics)
    return model


def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig("plots/loss_plot.png")
    plt.clf()


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
    plt.savefig("plots/metrics_plot.png")
    plt.clf()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("plots/confusion_matrix_plot.png")
    plt.clf()

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total: ', np.sum(cm[1]))


def plot_embbedings(model, lookup_dict, embedd):
    if embedd == "type":
        column_string = "combined_type"
    else:
        column_string = "Gen"
    layer_type = model.get_layer('embedding-%s' % embedd)
    output_embeddings_type = layer_type.get_weights()
    output_embeddings_type_df = pd.DataFrame(output_embeddings_type[0])
    output_embeddings_type_df = output_embeddings_type_df.reset_index()
    output_embeddings_type_df.columns = [column_string, 'embedding_1', 'embedding_2', 'embedding_3']
    m = output_embeddings_type_df.iloc[:, 1:].values
    labels = output_embeddings_type_df.iloc[:, 0:1].values
    if embedd == "type":
        def get_key(val):
            for key, value in lookup_dict.items():
                if val == value:
                    return key

        labels = [get_key(label) for label in labels]
    else:
        labels += 1
        labels = [y for x in labels for y in x]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(labels)):
        ax.scatter(m[i, 0], m[i, 1], m[i, 2], color='b')
        ax.text(m[i, 0], m[i, 1], m[i, 2], '%s' % (labels[i]), size=20, zorder=1, color='k')

    ax.set_xlabel('Embedding 1')
    ax.set_ylabel('Embedding 2')
    ax.set_zlabel('Embedding 3')
    plt.savefig("plots/3d_plot_embedding_%s.png" % embedd)
    plt.show()


def embedding_tensorboard(model, lookup_dict, embedd):
    # Set up a logs directory, so Tensorboard knows where to look for files
    log_dir = 'logs/%s' % embedd
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if embedd == "type":
        column_string = "combined_type"
    else:
        column_string = "Gen"
    layer_type = model.get_layer('embedding-%s' % embedd)
    output_embeddings_type = layer_type.get_weights()
    output_embeddings_type_df = pd.DataFrame(output_embeddings_type[0])
    output_embeddings_type_df = output_embeddings_type_df.reset_index()
    output_embeddings_type_df.columns = [column_string, 'embedding_1', 'embedding_2', 'embedding_3']
    m = output_embeddings_type_df.iloc[:, 1:].values
    labels = output_embeddings_type_df.iloc[:, 0:1].values
    if embedd == "type":
        def get_key(val):
            for key, value in lookup_dict.items():
                if val == value:
                    return key

        labels = [get_key(label) for label in labels]
    else:
        labels += 1
        labels = [y for x in labels for y in x]

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

    weights = tf.Variable(model.get_layer("embedding-%s" % embedd).get_weights()[0][0:])
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def main():
    # Read csv file
    lookup_dict = {}
    pokemon = pd.read_csv("data/pokemon.csv")
    pokemon["Type 2"] = pokemon["Type 2"].fillna("no_type")

    processed, lookup_dict = process_data(pokemon)

    x = processed.loc[:, ~processed.columns.isin(['Legendary'])].copy()
    y = processed["Legendary"].copy()

    # Split dataset to train, val and test set
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    embedding_size = 3

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
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    # Define checkpoints
    checkpoint_path = "model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    # Create model
    model = create_model(metrics, embedding_size)
    # Get summary
    model.summary()
    plot_model(model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)
    # ann_viz(model, filename="plots/neural_network.png", title="NN Architecture")

    # Get class weights
    neg, pos = np.bincount(pokemon["Legendary"])
    total = neg + pos
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    xval = [X_val[["combined_type"]], X_val["Gen"], X_val[NUM_ATTRIB]]
    # Train model
    training_history = model.fit(x=[X_train[["combined_type"]], X_train["Gen"], X_train[NUM_ATTRIB]],
                                 y=y_train, epochs=10000, batch_size=32, callbacks=[early_stopping, cp_callback],
                                 validation_data=(xval, y_val), class_weight=class_weight)

    # Plot loss
    plot_loss(training_history, "Training Loss", 0)

    # Plot metrics
    plot_metrics(training_history)

    # Perform evaluations on test set
    train_predictions_baseline = model.predict(x=[X_train[["combined_type"]], X_train["Gen"], X_train[NUM_ATTRIB]],
                                               batch_size=4)
    test_predictions_baseline = model.predict(x=[X_test[["combined_type"]], X_test["Gen"], X_test[NUM_ATTRIB]],
                                              batch_size=4)

    # Evaluate test set
    baseline_results = model.evaluate(x=[X_test[["combined_type"]], X_test["Gen"], X_test[NUM_ATTRIB]], y=y_test,
                                      batch_size=4, verbose=0)

    # Get metrics
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)

    # Plot confusion matrix
    plot_cm(y_test, test_predictions_baseline)

    # Plot ROC curve
    plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
    plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig("plots/roc_plot.png")

    # Get embeddings
    plot_embbedings(model, lookup_dict, "type")
    plot_embbedings(model, lookup_dict, "gen")

    # Get embeddings using Tensorboard
    embedding_tensorboard(model, lookup_dict, "type")
    embedding_tensorboard(model, lookup_dict, "gen")


if __name__ == "__main__":
    main()
