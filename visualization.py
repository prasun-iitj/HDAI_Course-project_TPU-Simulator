
import matplotlib.pyplot as plt
import numpy as np

def show_matrix(matrix,title="Matrix"):
    """We display matrices as heatmaps - it's much easier to see patterns this way!"""
    matrix=np.array(matrix)  # We convert to numpy array
    plt.imshow(matrix)  # We display as heatmap
    plt.title(title)  # We add the title
    plt.colorbar()  # We add a color scale to understand values
    plt.show()  # We show the plot
