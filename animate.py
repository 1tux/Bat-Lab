import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bat_class import Bat
from constants import *

from recordings import load_data

bats = [0] * N_BATS
x_data = [0] * N_BATS
y_data = [0] * N_BATS

def animation_frame(j):
    for i in range(N_BATS):
        bats[i].set_xdata(x_data[i][:j])
        bats[i].set_ydata(y_data[i][:j])
    return bats[i],

def main():
    fig, ax = plt.subplots()

    for i in range(N_BATS):
        x_data[i], y_data[i],_ = load_data(i)
        #display(x_data[i])
        #display(y_data[i])
        ax.set_xlim(MIN_X, MAX_X)
        ax.set_ylim(MIN_Y, MAX_Y)
        bats[i], = ax.plot(0, 0)

    animation = FuncAnimation(fig, func=animation_frame, frames=range(1000), interval=60)
    animation.save('tmp_animation.gif', writer='pillow', fps=20)
    plt.show()
    
