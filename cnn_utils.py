import matplotlib.pyplot as plt
import numpy as np


def plot_curve(training, test, graph_type):
    plt.plot(training)
    plt.plot(test)
    plt.margins()
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.grid(linestyle='dotted')
    plt.xticks(np.arange(0, len(training) + 1, 1.0))

    if graph_type == 'training':
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
    elif graph_type == 'loss':
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
