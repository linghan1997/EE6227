import random
import turtle
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def chess_board(li):
    turtle.tracer(0, 0)
    WIDTH = 40
    turtle.st()
    turtle.pu()
    turtle.goto(-len(li) / 2 * WIDTH, len(li) / 2 * WIDTH)
    turtle.pd()
    turtle.setup(width=len(li) * WIDTH + WIDTH, height=len(li) * WIDTH + WIDTH)
    turtle.title("Solving N-Queen...")

    def square(coloring):
        if coloring:
            turtle.begin_fill()
            for h in range(4):
                turtle.fd(WIDTH)
                turtle.rt(90)
            turtle.end_fill()
        else:
            for k in range(4):
                turtle.fd(WIDTH)
                turtle.rt(90)

    for i in range(1, len(li) + 1):
        for j in range(1, len(li) + 1):
            if ((i + j) % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                square(True)
                turtle.fd(WIDTH)
            else:
                square(False)
                turtle.fd(WIDTH)
        turtle.pu()
        turtle.goto(-len(li) / 2 * WIDTH, - WIDTH * i + len(li) / 2 * WIDTH)
        turtle.pd()
    turtle.pu()
    turtle.ht()

    for item in enumerate(li, 1):
        if item[0] <= len(li) / 2 and item[1] <= len(li) / 2:
            turtle.goto(-WIDTH * (len(li) / 2 + 1 - item[0]) + WIDTH / 2,
                        WIDTH * (len(li) / 2 + 1 - item[1]) - WIDTH / 2)
            turtle.dot(WIDTH / 2, 'red')
        if item[1] <= len(li) / 2 < item[0]:
            turtle.goto(WIDTH * (item[0] - len(li) / 2) - WIDTH / 2, WIDTH * (len(li) / 2 + 1 - item[1]) - WIDTH / 2)
            turtle.dot(WIDTH / 2, 'red')
        if item[0] <= len(li) / 2 < item[1]:
            turtle.goto(-WIDTH * (len(li) / 2 + 1 - item[0]) + WIDTH / 2, -WIDTH * (item[1] - len(li) / 2) + WIDTH / 2)
            turtle.dot(WIDTH / 2, 'red')
        if item[0] > len(li) / 2 and item[1] > len(li) / 2:
            turtle.goto(WIDTH * (item[0] - len(li) / 2) - WIDTH / 2, -WIDTH * (item[1] - len(li) / 2) + WIDTH / 2)
            turtle.dot(WIDTH / 2, 'red')

    turtle.mainloop()


class island:
    def __init__(self, bord_Size=8, crossoverProbability=0.5, mutationProbability=0.85):
        self.bord_size = bord_size
        self.populationSize = 2 * bord_Size
        self.crossoverProbability = crossoverProbability
        self.mutationProbability = mutationProbability
        self.cost_list = []
        self.population = []
        self.max_children = self.populationSize // 3
        self.max_offspring = 2

        # initializing PopulationSize chromosome
        for _ in range(self.populationSize):
            chromosome = list(range(1, self.bord_size + 1))
            random.shuffle(chromosome)
            chromosome.append(self.cost(chromosome))
            # e.x. chromosome = [4, 2, 6, 5, 3, 8, 1, 7, 3] cost = 3
            self.population.append(chromosome)

        # sorting Population with cost from small to large
        self.population.sort(key=lambda q: q[-1])
        # Population[0][-1]: the smallest cost
        self.cost_list.append(self.population[0][-1])

    def generation(self):
        random.shuffle(self.population)
        # recombine parents
        new_children = []
        for _ in range(self.max_children):
            p1, p2 = self.parent_selection(), self.parent_selection()
            done = False
            if random.random() < self.crossoverProbability:
                children = self.crossover(p1, p2)
                done = True
            else:
                # remain parents unchanged
                children = [p1[:], p2[:]]
            for child in children:
                if random.random() < self.mutationProbability or not done:
                    self.mutation(child)
                new_children.append(child)
        self.population.extend(new_children)

        # kill people with upper cost (goal : minimizing cost)
        self.population.sort(key=lambda q: q[-1])
        del self.population[self.populationSize:]

        self.cost_list.append(self.population[0][-1])

    # ----------------------utils for generations--------------------------#
    # parent selection function
    # tournament technique
    def parent_selection(self):
        tmp = (list(), self.bord_size)
        for _ in range(self.populationSize // 5):
            ch = random.choice(self.population)
            if ch[-1] < tmp[1]:
                tmp = (ch, ch[-1])
        return tmp[0]

    # cost function
    # return number of attacks
    def cost(self, chromosome):
        return sum([1 for i in range(self.bord_size) for j in range(i + 1, self.bord_size) if
                    abs(j - i) == abs(chromosome[j] - chromosome[i])])

    # crossover function
    # PMX technique
    def crossover(self, parent1, parent2):
        children = list()
        for _ in range(random.randint(1, self.max_offspring)):
            child = [-1] * (self.bord_size + 1)
            p, q = random.randint(1, self.bord_size // 2 - 1), random.randint(self.bord_size // 2 + 1,
                                                                              self.bord_size - 2)
            child[p: q + 1] = parent1[p: q + 1]
            for i in range(p, q + 1):
                if parent2[i] not in child:
                    t = i
                    while p <= t <= q:
                        t = parent2.index(parent1[t])
                    child[t] = parent2[i]
            for j in range(self.bord_size):
                if child[j] == -1:
                    child[j] = parent2[j]
            child[-1] = self.cost(child)
            children.append(child)
            parent1, parent2 = parent2, parent1
        return children

    # mutation function
    # single swap technique
    def mutation(self, chromosome):
        p, q = random.randint(0, self.bord_size - 1), random.randint(0, self.bord_size - 1)
        chromosome[p], chromosome[q] = chromosome[q], chromosome[p]
        chromosome[-1] = self.cost(chromosome)


if __name__ == "__main__":
    bord_size = 100
    exchange_size = 4  # numbers of individuals to exchange
    exchange_period = 25  # exchange individuals every ** generations

    islands = []
    island1 = island(bord_Size=bord_size, crossoverProbability=0.5, mutationProbability=0.85)
    islands.append(island1)
    island2 = island(bord_Size=bord_size, crossoverProbability=0.6, mutationProbability=0.75)
    islands.append(island2)
    island3 = island(bord_Size=bord_size, crossoverProbability=0.4, mutationProbability=0.65)
    islands.append(island3)

    iteration_count = 0

    start_time = datetime.now()

    # iteration ends if any island has achieved a feasible solution for N-Queen
    while island1.population[0][-1] and island2.population[0][-1] and island3.population[0][-1]:
        # each island produces its population separately (in parallel if on a parallel machine)
        for i in range(len(islands)):
            islands[i].generation()

        iteration_count += 1

        # perform individual exchange among islands
        if iteration_count == exchange_period:
            exchange_pairs = [[0, 1], [0, 2], [1, 2]]
            for pair in exchange_pairs:
                island_1 = islands[pair[0]]
                island_2 = islands[pair[1]]
                island_1_population = np.asarray(island_1.population)
                island_2_population = np.asarray(island_2.population)
                immigrants_idx = np.random.randint(0, island_1.populationSize, exchange_size)
                immigrants = island_1_population[immigrants_idx]
                island_1_population[immigrants_idx] = island_2_population[immigrants_idx]
                island_2_population[immigrants_idx] = immigrants
                island_1.population = island_1_population.tolist()
                island_1.population.sort(key=lambda q: q[-1])
                island_2.population = island_2_population.tolist()
                island_2.population.sort(key=lambda q: q[-1])

        if iteration_count % 10 == 0:
            print("%dth iteration, current cost: %d" % (iteration_count, min(island1.population[0][-1],
                                                                             island2.population[0][-1],
                                                                             island3.population[0][-1])))

    end_time = datetime.now()

    index = np.argmin([island1.population[0][-1], island2.population[0][-1], island3.population[0][-1]])

    del islands[index].population[0][-1]
    print("%dth iteration, current cost: %d" % (iteration_count, 0))
    print(
        "solution for %d Queen is: %s" % (
            bord_size, str([pair[::-1] for pair in enumerate(islands[index].population[0], 1)])))
    print("total time for solution is %s" % (str(end_time - start_time)))

    # print the cost of the island which achieved a feasible solution
    iteration = range(len(islands[index].cost_list))
    plt.plot(iteration, islands[index].cost_list)
    plt.grid(True)

    if len(island1.cost_list) > 0:
        plt.ylim((0, max(islands[index].cost_list) + 1))
        plt.ylabel('cost value (number of attacks)')
        plt.xlabel('iteration')
        plt.title(str(str(bord_size) + "-Queen solved in " + str(end_time - start_time)))
        plt.show()

    # draw on chess_board
    if bord_size <= 16:
        chess_board(islands[index].population[0])
