import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import numpy as np
from pyswarms.utils.functions import single_obj as fx
from tuning.function import custom_fitness

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

min_x = np.array([1, 2, 0, 2, 0, 0.0001, 0.0001, 1, 4])
max_x = np.array([8, 64, 0.5, 64, 0.5, 0.1, 0.1, 5, 32])
bounds = (min_x, max_x)
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10,
                                    dimensions=len(min_x),
                                    bounds=bounds,
                                    options=options)
# Perform optimization
best_cost, best_pos = optimizer.optimize(custom_fitness,
                                         iters=100,
                                         n_processes=6)

plot_cost_history(optimizer.cost_history)
plt.show()



