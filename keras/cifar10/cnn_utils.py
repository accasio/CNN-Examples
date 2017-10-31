import matplotlib
import sys
from keras.utils import get_file
from zmq.backend.cython.socket import cPickle

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_curve(training, test, graph_type):
    plt.plot(training)
    plt.plot(test)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.grid(linestyle='dotted')
    plt.xticks(np.arange(0, len(training) + 1, 1.0))

    if 'accuracy' in graph_type:
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
    elif 'loss' in graph_type:
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig('/output/' + graph_type)
    plt.clf()


def plot_scatter_density(x, y, graph_type):
    plt.grid(linestyle='dotted')
    plt.plot(x, y, marker='o')
    if 'accuracy' in graph_type:
        plt.title('Model Accuracy vs Density of fully connected layer')
        plt.ylabel('Accuracy')
        plt.xlabel('Density')
    elif 'loss' in graph_type:
        plt.title('Model Loss vs Density of fully connected layer')
        plt.ylabel('Loss')
        plt.xlabel('Density')

    for i, j in zip(x, y):
        plt.text(i, j, '(' + str(i) + ', ' + str(j) + ')')

    plt.savefig('/output/' + graph_type)
    plt.clf()


def plot_scatter_filters(x, y, graph_type):
    plt.grid(linestyle='dotted')
    plt.plot(x, y, marker='o')
    if 'accuracy' in graph_type:
        plt.title('Model Accuracy vs Filters in 1st & 2nd conv layers')
        plt.ylabel('Accuracy')
        plt.xlabel('# of filters')
    elif 'loss' in graph_type:
        plt.title('Model Loss vs Filters in 1st & 2nd conv layers')
        plt.ylabel('Loss')
        plt.xlabel('# of filters')

    for i, j in zip(x, y):
        plt.text(i, j, '(' + str(i) + ', ' + str(j) + ')')

    plt.savefig('/output/' + graph_type)
    plt.clf()


def plot_model_history(model_history, density):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc'])+1), len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig('/output/accuracy_and_loss_' + str(density))
    plt.clf()

