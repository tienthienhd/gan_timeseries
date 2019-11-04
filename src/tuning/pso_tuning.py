import random
import math


# +++++++++++ Cost function ++++++++++++++
def function1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2

    return total


num_dimensions = 1


class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle position
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, cost_func):
        self.err_i = cost_func(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_best_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight(how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant
        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimun position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO:
    def __init__(self, cost_func, x0, bounds, num_particles, max_iter):
        global num_dimensions
        num_dimensions = len(x0)

        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        for i in range(0, max_iter):
            # cycle through particles in swarm and evaluate fitness
            for j in range(num_particles):
                swarm[j].evaluate(cost_func)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through warm and update velocities and position
            for j in range(num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)

        print("FINAL:")
        print(pos_best_g)
        print(err_best_g)


if __name__ == '__main__':
    initial = [5, 5]  # initial starting location [x1, x2, ...]
    bounds = [(-10, 10), (-10, 10)] # input bounds [(x1min, x1max), (x2min, x2max)]
    PSO(function1, initial, bounds, num_particles=150, max_iter=300)



domain = [
        {'name': 'n_in', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
        {'name': 'n_out', 'type': 'discrete', 'domain': [1]},
        {'name': 'g_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
        {'name': 'g_dropout', 'type': 'continuous', 'domain': (0, 0.8)},

        {'name': 'd_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
        {'name': 'd_dropout', 'type': 'continuous', 'domain': (0, 0.8)},
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
        {'name': 'num_train_d', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
        {'name': 'batch_size', 'type': 'discrete', 'domain': [4, 8, 16, 32]},
    ]

def parse_space(space):
    pass