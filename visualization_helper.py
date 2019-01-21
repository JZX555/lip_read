# encoding=utf8
# import numpy as np
from matplotlib import pyplot as plt
import sys


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def percent(current, total):
    """
        just for fun
    """
    if current == total - 1:
        current = total
    bar_length = 20

    hashes = '#' * int(current / total * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write(
        "\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total)))
