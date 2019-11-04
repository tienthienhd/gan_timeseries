import random
import numpy as np
import multiprocessing as mp


class Particle:
    def __init__(self, type_attr, min_val, max_val, w, c1, c2):
        self.type_attr = type_attr
        self.min_val = min_val
        self.max_val = max_val

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.position = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(type_attr))
        self.position = self._corect_pos(self.position)

        self.velocity = np.random.uniform(-1, 1, len(type_attr))  # particle position
        self.pos_best = self.position  # best position individual

        self.err = -1  # error individual
        self.err_best = -1  # best error individual

    def _corect_pos(self, position):
        for i, t in enumerate(self.type_attr):
            if t == 'discrete':
                position[i] = int(position[i])
        return position

    # evaluate current fitness
    def evaluate(self, cost_function):
        self.err = cost_function(self.position)

        # check to see if the current position is an individual best
        if self.err < self.err_best or self.err_best == -1:
            self.pos_best = self.position
            self.err_best = self.err

    def update_velocity(self, pos_bess_g):
        r1 = np.random.rand(len(self.type_attr))
        r2 = np.random.rand(len(self.type_attr))

        vel_cognitive = self.c1 * r1 * (self.pos_best - self.position)
        vel_social = self.c2 * r2 * (pos_bess_g - self.position)
        self.velocity = self.w * self.velocity + vel_cognitive + vel_social

    def update_position(self):
        self.position = self.position + self.velocity
        self.position = self._corect_pos(self.position)
        self.position = np.clip(self.position, self.min_val, self.max_val)


class PSO:
    def __init__(self, fn, domain, num_particles):
        self.cost_function = fn
        self._parse_domain(domain)
        self.num_particles = num_particles
        self.num_dimensions = len(domain)

        self._create_particles()

        self.err_best_g = -1
        self.pos_best_g = None

    def _parse_domain(self, domain):
        name = []
        type_attr = []
        max_val = []
        min_val = []
        range_val = []
        for attr in domain:
            name.append(attr['name'])
            type_attr.append(attr['type'])
            if attr['type'] == 'discrete':
                min_val.append(0)
                max_val.append(len(attr['domain']))
            elif attr['type'] == 'continuous':
                min_val.append(attr['domain'][0])
                max_val.append(attr['domain'][1])
            range_val.append(attr['domain'])

        self.name = name
        self.type_attr = type_attr
        self.min_val = np.array(min_val)
        self.max_val = np.array(max_val)
        self.range_val = range_val

    def decode_position(self, position):
        result = {}
        for i, t in enumerate(self.type_attr):
            if t == 'discrete':
                result[self.name[i]] = self.range_val[i][int(position[i])]
            else:
                result[self.name[i]] = position[i]

        return result

    def _create_particles(self):
        self.particles = []
        for i in range(self.num_particles):
            self.particles.append(Particle(self.type_attr, self.min_val, self.max_val, w=0.6, c1=1.2, c2=1.2))

    def run(self, max_iter):
        w_max = 0.9
        w_min = 0.4
        # begin optimization loop
        for i in range(max_iter):
            w = (max_iter - i) / max_iter * (w_max - w_min) + w_min
            # update w after each move (weight down)
            for particle in self.particles:
                particle.w = w

            # cycle through particles in swarm and evaluate fitness
            for particle in self.particles:
                pool = mp.Pool()
                particle.evaluate(self.cost_function)

                # determine if current particle is the best (globally)
                if particle.err < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = particle.position
                    self.err_best_g = particle.err

            # cycle through warm and update velocities and position
            for particle in self.particles:
                particle.update_velocity(self.pos_best_g)
                particle.update_position()

            # print(self.err_best_g)

        print("FINAL")
        print(self.pos_best_g)
        print(self.err_best_g)
        print(self.decode_position(self.pos_best_g))


if __name__ == '__main__':

    def fitness(x):
        total = 0
        for i in range(len(x)):
            total += x[i] ** 2

        return total


    domain = [
        {'name': 'x1', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x2', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x3', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x4', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x5', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x6', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x7', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x8', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x9', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x10', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x11', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x12', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x13', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x14', 'type': 'continuous', 'domain': (-100, 100)},
        {'name': 'x15', 'type': 'continuous', 'domain': (-100, 100)},
        # {'name': 'x16', 'type': 'continuous', 'domain': (-100, 100)},
        # {'name': 'x17', 'type': 'continuous', 'domain': (-100, 100)},
        # {'name': 'x18', 'type': 'continuous', 'domain': (-100, 100)},
        # {'name': 'x19', 'type': 'continuous', 'domain': (-100, 100)},
        # {'name': 'x20', 'type': 'continuous', 'domain': (-100, 100)},
    ]

    from tuning.metaheuristic.function_utils import *
    a = PSO(C30, domain, 1000)
    a.run(100)

    # def test(input_x):
    #     return C30(input_x[0])
    #
    # from GPyOpt.methods import BayesianOptimization
    #
    # bayes = BayesianOptimization(test, domain, initial_design_numdata=100, num_cores=-1)
    # bayes.run_optimization(10, verbosity=False)
    # bayes.plot_convergence()
    # bayes.plot_acquisition()
    # print(bayes.x_opt)
    # print(bayes.Y_best[-1])
