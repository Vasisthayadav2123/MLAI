from two_layer import forward_pass, X, y
import math
import random

def fitness(genome):
    total_loss = 0

    for x_i, y_i in zip(X, y):
        a1, a2, p = forward_pass(x_i, genome)

        # Protect against log(0)
        p = max(1e-7, min(1 - 1e-7, p))

        # Binary cross-entropy
        loss = -(y_i * math.log(p) +
                 (1 - y_i) * math.log(1 - p))

        total_loss += loss

    return total_loss / len(X)


def select_best(population, n , fitness):
    best_genomes = sorted(population, key=fitness)
    return best_genomes[:n]

def create_population(size):
    return [
        [random.uniform(-10, 10) for _ in range(9)]
        for _ in range(size)
    ]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutate(genome):
    new_genome = genome.copy()

    for i in range(len(new_genome)):
        if random.random() < 0.1:
            new_genome[i] += random.gauss(0, 0.5)

    return new_genome



def evolve(population, fitness, generations):

    for gen in range(generations):

        best_genomes = select_best(population, 10, fitness)

        children = []

        while len(children) < 40:
            parent1, parent2 = random.sample(best_genomes, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            children.append(child)

        population = best_genomes + children

        if gen % 10 == 0:
            best_score = fitness(best_genomes[0])
            print(f"Generation {gen}, Best Score: {best_score}")

    return population
        
    
if __name__ == "__main__":
    population = create_population(50)
    population = evolve(population, fitness, 300)
    winner = min(population, key=fitness)
    print("Final loss:", fitness(winner))
    for x_i, y_i in zip(X, y):
        a1, a2, p = forward_pass(x_i, winner)
        print(f"{x_i}  target={y_i}  a1={a1:.3f}  a2={a2:.3f}  p={p:.3f}")
    