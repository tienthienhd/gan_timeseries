import random
import numpy as np
import multiprocessing as mp


class Particle:
    def __init__(self, type_attr, min_val, max_val, range_val, w, c1, c2):
        self.type_attr = type_attr
        self.min_val = min_val
        self.max_val = max_val
        self.range_val = range_val

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

    def decode_position(self, position):
        result = []
        for i, t in enumerate(self.type_attr):
            if t == 'discrete':
                result.append(self.range_val[i][int(position[i])])
            else:
                result.append(position[i])

        return result

    # evaluate current fitness
    def evaluate(self, cost_function):
        self.err = cost_function(self.decode_position(self.position))

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
                max_val.append(len(attr['domain']) - 1)
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
            self.particles.append(
                Particle(self.type_attr, self.min_val, self.max_val, self.range_val, w=0.6, c1=1.2, c2=1.2))

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
                particle.evaluate(self.cost_function)

                # determine if current particle is the best (globally)
                if particle.err < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = particle.position.copy()
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


from tuning.config import domain
from tuning.function import fitness_function

a = PSO(fitness_function, domain, num_particles=20)
a.run(10)
