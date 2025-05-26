#PYTHON
#PYTHON
import numpy as np


class Problem():
    def __init__(self, num_n, num_k, height_n, width_n, width_k, height_k, cost_k):
        self.num_n = num_n
        self.num_k = num_k
        self.height_n = height_n
        self.width_n = width_n
        self.width_k = width_k
        self.height_k = height_k
        self.cost_k = cost_k


class Individual():
    def __init__(self):
        self.chromosome = None
        self.fitness = None

    def gen_chromosome(self, num_n, num_k):
        self.chromosome = np.random.permutation(num_n).tolist()

    def decode(self, decode_function, problem):
        return decode_function(self.chromosome, problem)

    def cal_fitness(self, decode_function, cal_metric, problem):
        solution = self.decode(decode_function, problem)
        self.fitness = cal_metric(solution, problem)


class Population():
    def __init__(self, pop_size, problem):
        self.pop_size = pop_size
        self.problem = problem
        self.indi_list = []

    def gen_pop(self, cal_metric, decode_function):
        while len(self.indi_list) < self.pop_size:
            indi = Individual()
            indi.gen_chromosome(self.problem.num_n, self.problem.num_k)
            indi.cal_fitness(decode_function, cal_metric, self.problem)
            if indi.fitness != float('inf'):
                self.indi_list.append(indi)


def crossover(parent1: Individual, parent2: Individual):
    size = len(parent1.chromosome)
    cut = np.random.randint(1, size)
    off1 = Individual()
    off2 = Individual()
    off1.chromosome = parent1.chromosome[:cut] + parent2.chromosome[cut:]
    off2.chromosome = parent2.chromosome[:cut] + parent1.chromosome[cut:]
    return off1, off2


def mutation(parent: Individual, num_k):
    off = Individual()
    off.chromosome = parent.chromosome[:]
    i = np.random.randint(len(off.chromosome))
    j = np.random.randint(len(off.chromosome))
    off.chromosome[i], off.chromosome[j] = off.chromosome[j], off.chromosome[i]
    return off

#greddy fit
def decode_function(chromosome, problem):
    placements = [None] * problem.num_n
    placed_items = {k: [] for k in range(problem.num_k)}
    sorted_trucks = sorted(range(problem.num_k), key=lambda k: problem.cost_k[k])

    for item_index in chromosome:
        w, h = problem.width_n[item_index], problem.height_n[item_index]
        placed = False

        for k in sorted_trucks:
            Wk, Hk = problem.width_k[k], problem.height_k[k]
            for rot in [0, 1]:
                iw, ih = (w, h) if rot == 0 else (h, w)
                step = 5
                for x in range(0, Wk - iw + 1, step):
                    for y in range(0, Hk - ih + 1, step):
                        overlap = False
                        for (xi, yi, wi, hi) in placed_items[k]:
                            if not (x + iw <= xi or x >= xi + wi or y + ih <= yi or y >= yi + hi):
                                overlap = True
                                break
                        if not overlap:
                            placements[item_index] = (item_index, k + 1, x, y, rot)
                            placed_items[k].append((x, y, iw, ih))
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return placements


def cal_metric(solution, problem):
    if solution is None or any(s is None for s in solution):
        return float('inf')
    # print(solution)
    used_trucks = {k for _, k, _, _, _ in solution}
    return sum(problem.cost_k[k - 1] for k in used_trucks)


def ga(problem, num_gen, pop_size, pc, pm, decode_function, cal_metric, max_min="min"):
    pop = Population(pop_size, problem)
    pop.gen_pop(cal_metric, decode_function)

    for gen in range(num_gen):
        off_list = []
        for _ in range(pop_size // 2):
            parent1, parent2 = np.random.choice(pop.indi_list, 2, replace=False)
            if np.random.rand() <= pc:
                off1, off2 = crossover(parent1, parent2)
            else:
                off1, off2 = parent1, parent2
            if np.random.rand() <= pm:
                off1 = mutation(off1, problem.num_k)
            if np.random.rand() <= pm:
                off2 = mutation(off2, problem.num_k)

            off1.cal_fitness(decode_function, cal_metric, problem)
            off2.cal_fitness(decode_function, cal_metric, problem)

            if off1.fitness != float('inf'):
                off_list.append(off1)
            if off2.fitness != float('inf'):
                off_list.append(off2)

        pop.indi_list.extend(off_list)
        pop.indi_list = [indi for indi in pop.indi_list if indi.fitness != float('inf')]
        if not pop.indi_list:
            print("Không còn cá thể hợp lệ trong thế hệ.")
            return None
        pop.indi_list.sort(key=lambda x: x.fitness, reverse=(max_min == "max"))
        pop.indi_list = pop.indi_list[:pop_size]

    return pop.indi_list[0] if pop.indi_list else None


if __name__ == "__main__":
    # Input
    num_n, num_k = map(int, input().split())
    width_n, height_n = [], []
    for _ in range(num_n):
        a, b = map(int, input().split())
        width_n.append(a)
        height_n.append(b)

    width_k, height_k, cost_k = [], [], []
    for _ in range(num_k):
        a, b, c = map(int, input().split())
        width_k.append(a)
        height_k.append(b)
        cost_k.append(c)

    Bin_packing = Problem(num_n, num_k, height_n, width_n, width_k, height_k, cost_k)

    best = ga(
        Bin_packing,
        num_gen=10,
        pop_size=10,
        pc=0.8,
        pm=0.2,
        decode_function=decode_function,
        cal_metric=cal_metric
    )

    if best is None:
        print("Không tìm được lời giải hợp lệ.")
    else:
        print("Best fitness:", best.fitness)
        sol = decode_function(best.chromosome, Bin_packing)
        for i, t, x, y, o in sol:
            print(i + 1, t, x, y, o)
