
import matplotlib.pyplot as plt
import numpy as np
import time

def animate_array(n=3):

    grid=np.zeros((n,n))

    for cycle in range(5):

        grid=np.random.randint(0,9,(n,n))

        plt.imshow(grid)
        plt.title(f"TPU Systolic Array Cycle {cycle+1}")
        plt.colorbar()
        plt.pause(0.8)
        plt.clf()

    plt.close()

if __name__=="__main__":
    animate_array()
