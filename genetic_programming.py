import pandas as pd
import numpy as np
import math
import copy
from decision_tree import *


class Individual:
    def __init__(self, features):
        self.features = features
        self.chromosome = decision_tree(features)
        self.fitness = 0.0

    def __lt__(self, other):
        return self.fitness >= other.fitness

    def get_accuracy(self, data):
        means_vector = data[data['class'] == 1][self.features].mean()

        all_instances = data.shape[0]
        correct = 0

        for i in range(all_instances):
            predicted = self.chromosome.classify_mail(data.iloc[i ,:], means_vector)
            true_class = int(data.iloc[i ,:]['class'])

            if predicted == true_class:
                correct += 1

        return correct / all_instances

    def calculate_fitness(self, data):
        self.fitness = self.get_accuracy(data)
        return self.fitness

    def __str__(self):
        return str(self.fitness) + " "


def initialize_population(features, population_size):
    population = [None] * population_size

    for i in range(population_size):
        population[i] = Individual(features)

    return population


def mutate(individual):
    position = np.random.randint(0, 15)
    depth = np.floor(np.log2(position + 1))
    individual.chromosome.tree[position] = np.random.randint(0, len(individual.features) - depth)

    return individual


def get_positions_in_subtree__(position, positions):
    positions.append(2 * position + 1)
    positions.append(2 * position + 2)

    if 2 * position + 1 < 15:
        positions = get_positions_in_subtree__(2 * position + 1, positions)
        positions = get_positions_in_subtree__(2 * position + 2, positions)

    return positions


def get_positions_in_subtree(position):
    positions = [position]

    if position < 15:
        positions = get_positions_in_subtree__(position, positions)

    return positions


def cross_over(individual1, individual2, child1, child2):
    position = np.random.randint(0, 15)
    positions_in_subtree = get_positions_in_subtree(position)

    for i in range(0, 31):
        if i in positions_in_subtree:
            child1.chromosome.tree[i] = individual2.chromosome.tree[i]
            child2.chromosome.tree[i] = individual1.chromosome.tree[i]
        else:
            child1.chromosome.tree[i] = individual1.chromosome.tree[i]
            child2.chromosome.tree[i] = individual2.chromosome.tree[i]


def selection(population):
    total_fitness = sum([individual.fitness for individual in population])
    fitness_proportions = [(individual.fitness / total_fitness) for individual in population]

    return np.random.choice(population, p=fitness_proportions)

