from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def learning_curve(x,ys,filename='learning_curve.png'):
    ysnp = np.array(ys)
    print(np.shape(ysnp))
    fig, axs = plt.subplots(len(ys),figsize=(20,10))
    for i in range(len(ys)):
        axs[i].axis('auto')
        axs[i].set(xlabel='iterations', ylabel='V[{0}]' . format(i))
        line = axs[i].plot(x,ys[i],color='k')
    fig.savefig(filename)
