import click
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

flatten = lambda l: [item for sublist in l for item in sublist]


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def parseXml(file_path):
    try:
        root = ET.parse(file_path).getroot()
        name = root.find('name').text
        description = root.find('description').text
        graphNode = root.find('graph')
        graph = {}
        for index, vertexNode in enumerate(graphNode, start=0):
            vertex = {}
            for edgeNode in vertexNode:
                attrib = edgeNode.attrib
                cost = attrib['cost']
                label = int(edgeNode.text)
                vertex[label] = float(cost)
            graph[index] = vertex
        vertexNumber = len(graph)
        return name, description, graph, vertexNumber

    except Exception as e:
        print(e)


def get_first_generations(size, len):
    return [random.sample(range((len)), len) for x in range(0, size)]


def crossover_func_pm(gene1, gene2, len):
    cut_one = random.randint(1, len / 2)
    cut_two = random.randint(cut_one, len - 1)
    # cut_one, cut_two = [(cut_one, cut_two) if cut_one < cut_two else cut_two, cut_one]
    # print(cut_one, cut_two)
    off1 = [-1] * len
    off2 = [-1] * len

    off1_miss = diff(gene1[cut_one:cut_two], gene2[cut_one:cut_two])
    off2_miss = diff(gene2[cut_one:cut_two], gene1[cut_one:cut_two])

    for i in range(cut_one, cut_two):
        off1[i] = gene2[i]
        off2[i] = gene1[i]

    for i in range(0, cut_one):
        if not gene1[i] in off1:
            off1[i] = gene1[i]
        else:
            off1[i] = off1_miss.pop()
        if not gene2[i] in off2:
            off2[i] = gene2[i]
        else:
            off2[i] = off2_miss.pop()

    for i in range(cut_two, len):
        if not gene1[i] in off1:
            off1[i] = gene1[i]
        else:
            off1[i] = off1_miss.pop()
        if not gene2[i] in off2:
            off2[i] = gene2[i]
        else:
            off2[i] = off2_miss.pop()
    return off1, off2


def get_fitness_func(graph, vertexNuber):
    def fitness(way):
        try:
            if max(way) > vertexNuber - 1:
                raise Exception('Way vertex {} are invalid'.format(max(way)))
            if len(way) < vertexNuber:
                raise Exception('Used way not pass trough all vertex'.format(len(way)))
            cost = 0
            for edge in list(filter(lambda x: len(x) == 2, [way[i:i + 2] for i in range(0, len(way), 1)])):
                vert = graph[edge[0]]
                edge_cost = vert[edge[1]]
                cost += edge_cost
            return cost
        except Exception as e:
            print(e)

    return fitness


def selection(population, size, fitness):
    res = sorted(population, key=lambda x: fitness(x))
    return res[:size], res[size:]


def crossover(population, crossover_func, chromo_size):
    return flatten([crossover_func(ch1, ch2, chromo_size) for ch1, ch2 in
                    list(filter(lambda x: len(x) == 2, [population[i:i + 2] for i in range(0, len(population), 2)]))])


def mutation(population, vertex_number, mutation_rate):
    try:
        sel = random.sample(range(len(population)), mutation_rate)
        result = population[:]
        for index in sel:
            to_change = random.sample(range(vertex_number - 1), 2)
            selected_chromo = population[index]
            mutated = selected_chromo[:]
            mutated[to_change[0]] = selected_chromo[to_change[1]]
            mutated[to_change[1]] = selected_chromo[to_change[0]]
            result[index] = mutated
        return result
    except Exception as e:
        print(e)


def aggregate(arr1, arr2):
    if np.array_equal(arr1, arr2):
        return arr1
    else:
        return arr1 + arr2


def ga(graph, generations, chromosomes_size, selection_size, mutation_rate, vertex_number):
    print(graph)
    fitness_func = get_fitness_func(graph, vertex_number)
    population = get_first_generations(int(chromosomes_size), vertex_number)
    best_chromo = population[0]
    best_fit = 0
    for generation in range(0, generations):
        print('Generation {}'.format(generation))
        try:
            best_fit = fitness_func(best_chromo)
            selected, remain = selection(population, selection_size, fitness_func)
            crossed = crossover(selected, crossover_func_pm, vertex_number)
            aggregated = aggregate(selected, crossed)
            mutated = mutation(aggregated, vertex_number, mutation_rate)
            rest = (len(population) - (len(mutated)))
            population = mutated + remain[:rest]
            for chromo in population:
                fit = fitness_func(chromo)
                print(fit, chromo)
                if fit < best_fit:
                    best_chromo = chromo
        except Exception as e:
            print(e)
            raise
    return best_fit,best_chromo
    # cost = fittnes_func([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 14, 13, 11, 12, 10, 0])
    # print(cost)
    # crossover_func_pm([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 14, 13, 11, 12, 10, 0],
    #                   [3, 4, 5, 6, 7, 8, 9, 15, 13, 11, 12, 10, 0, 1, 2, 14], vertexNumber)


@click.command()
@click.option('--generations', default=1, help='Number of generations.')
@click.option('--init-size', prompt='Give starting chromosomes group size',
              help='Size of starting chromosome group.')
@click.option('--selection-size', prompt='Give selection size',
              help='The selection size.')
@click.option('--mutation-rate', prompt='Give mutation rate',
              help='The mutation rate.')
@click.option('--input-file', prompt='Give input file',
              help='The data file.')
@click.option('--test-repetitions', prompt='Give execution repetition number',
              help='How may time to repeat execution.')
def salesman(generations, init_size, selection_size, mutation_rate, input_file, test_repetitions):
    name, description, graph, vertex_number = parseXml(input_file)
    print('File name: {}'.format(name))
    print('Data descriptions: {}'.format(description))
    print('Number of generations: {}'.format(generations))

    test_results = []
    chromo = []
    for index in range(1, int(test_repetitions)+1):
        best_fit, best_chromo = ga(graph, int(generations), int(init_size), int(selection_size), int(mutation_rate), vertex_number)
        test_results.append(best_fit)
        chromo.append(best_chromo)
    bestExecution = min(test_results)
    print('Best execution score {}, path {}'.format(bestExecution, chromo[test_results.index(bestExecution)]))
    worstExecution = max(test_results)
    print('Worst execution score {}, path {}'.format(worstExecution, chromo[test_results.index(worstExecution)]))
    average = np.average(test_results)
    print('Average execution score {}'.format(average))
    plt.plot(test_results)
    plt.ylabel('Results')
    plt.show()

if __name__ == '__main__':
    salesman()
