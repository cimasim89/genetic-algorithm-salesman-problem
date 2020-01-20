import click
import time
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import copy

VERSION = '1.0.1.1'


def flatten(l):
    return [item for sublist in l for item in sublist]


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def parse_xml(file_path):
    """
    La funzione prende come input il file xml che descrive la topologia della rete
    utilizzando la struttura definita dai file scaricati da TSPLIB e ne estrae
    nome, descrizione, grafo e il numero delle stazioni che lo compongono
    :param file_path: string path del file
    :return:
        name: string,
        description: string,
        graph: dict,
        vertex_number: int
    """
    try:
        root = ET.parse(file_path).getroot()
        name = root.find('name').text
        description = root.find('description').text
        graph_node = root.find('graph')
        graph = {}
        for index, vertexNode in enumerate(graph_node, start=0):
            vertex = {}
            for edgeNode in vertexNode:
                attrib = edgeNode.attrib
                cost = attrib['cost']
                label = int(edgeNode.text)
                vertex[label] = float(cost)
            graph[index] = vertex
        vertex_number = len(graph)
        return name, description, graph, vertex_number

    except Exception as e:
        print(e)


def get_first_generations(size, length):
    """
    Questa funzione crea la prima generazione random
    :param size: int
    :param length: int
    :return: [[int]]
    """
    return [random.sample(range(length), length) for _ in range(0, size)]


def crossover_func_pm(gene1, gene2, length):
    """
    Funzione di crossover
    Dati due geni selezionati genera due nuovi geni.
    L' algoritmo sceglie random due indici, uno nella prima meta il secondo nello spazio rimanente
    questi saranno utilizzati per effetture la combinazione dei due geni genitori.
    :param gene1: [int] genitore 1
    :param gene2: [int] genitore 2
    :param length: integer dimensione
    :return:
        figlio1: [int],
        figlio2: [int]
    """
    cut_one = random.randint(1, length // 2)
    cut_two = random.randint(cut_one + 1, length - 1)
    off1 = [-1] * length
    off2 = [-1] * length

    off1_miss = diff(gene1[cut_one:cut_two], gene2[cut_one:cut_two])
    off2_miss = diff(gene2[cut_one:cut_two], gene1[cut_one:cut_two])

    for i in range(cut_one, cut_two):
        off1[i] = gene2[i]
        off2[i] = gene1[i]

    def filler(start, end):
        for k in range(start, end):
            if not gene1[k] in off1:
                off1[k] = gene1[k]
            else:
                off1[k] = off1_miss.pop()
            if not gene2[k] in off2:
                off2[k] = gene2[k]
            else:
                off2[k] = off2_miss.pop()

    filler(0, cut_one)
    filler(cut_two, length)
    return off1, off2


def get_fitness_func(graph, vertex_number):
    """
    Questa funzione fornisce la funzione di fitness basata sul grafo corrente
    :param graph: dict
    :param vertex_number: string
    :return: [int] -> integer
    """

    def fitness(way):
        try:
            if max(way) > vertex_number - 1:
                raise Exception('Way vertex {} are invalid'.format(max(way)))
            if len(way) < vertex_number:
                raise Exception('Used way not pass trough all vertex'.format(len(way)))
            cost = 0
            edges = list(filter(lambda x: len(x) == 2, [way[i:i + 2] for i in range(0, len(way), 1)]))
            edges.append([way[-1], way[0]])
            for edge in edges:
                vert = graph[edge[0]]
                edge_cost = vert[edge[1]]
                cost += edge_cost
            return cost
        except Exception as e:
            print(e)

    return fitness


def selection(population, size, fitness):
    """
    Funzione di selezione
    data una [population] di individui una funzione di [fitness] seleziona
    [size} elementi dove il valore restituito da fitness e' minore
    :param population:  [[int]]
    :param size: int
    :param fitness: [int] -> int
    :return:
    """
    res = sorted(population, key=lambda x: fitness(x))
    return res[:size]


def generate_random_pairs(population_size, size):
    """
    Data la dimensione delle popolazione denera tante coppie quante richieste
    :param population_size:
    :param size:
    :return:
    """
    pairs = []
    for p in range(size):
        tmp = list(range(0, population_size))
        pairs.append((tmp.pop(random.randrange(len(tmp))),
                      tmp.pop(random.randrange(len(tmp)))))
    return pairs


def crossover(population, crossover_func, chromo_size, original_population_size):
    """
    Esecuzione della [crossover_func] sulla [population] selezionata
    :param population: [[int]]
    :param crossover_func: [int],[int],int -> [int],[int}
    :param chromo_size:
    :param original_population_size:
    :return:
    """
    return flatten([crossover_func(population[ch1], population[ch2], chromo_size)
                    for ch1, ch2 in generate_random_pairs(len(population), original_population_size)])


def get_roluette_wheel_pair(population, fitness_func, size):
    pairs = []
    population = [{'fitness': (fitness_func(chromosome)), 'chromosome': chromosome} for chromosome in population]
    for p in range(size):
        chromosomes = copy.copy(population)
        sel1, chromosomes = weighted_random_choice(chromosomes)
        sel2, chromosomes = weighted_random_choice(chromosomes)
        pairs.append((sel1, sel2))
    return pairs


def weighted_random_choice(population):
    max_fit = sum(chromosome.get('fitness') for chromosome in population)

    def mapper(x):
        x['probability'] = max_fit / x.get('fitness')
        return x

    chromosomes = list(map(mapper, population))
    max_probability = sum(chromosome.get('probability') for chromosome in chromosomes)
    pick = random.uniform(0, max_probability)
    current = 0
    rest = []
    selected = None
    for chromosome in chromosomes:
        current += chromosome.get('probability')
        if current > pick and selected is None:
            selected = chromosome
        else:
            rest.append(chromosome)
    return selected.get('chromosome'), rest


def roulette_crossover(population, fitness_func, crossover_func, chromo_size, original_population_size):
    return flatten([crossover_func(ch1, ch2, chromo_size)
                    for ch1, ch2 in get_roluette_wheel_pair(population, fitness_func, original_population_size)])


def mutation(population, vertex_number, mutation_rate):
    """
    Da una popolazione esegue su [mutation_rate] individui random una mutazione
    :param population: [[int]]
    :param vertex_number: int
    :param mutation_rate: int
    :return:
    """
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


def ga(graph, generations, chromosomes_number, selection_size, roulette, mutation_rate, vertex_number):
    """
    Esecuzione dell'algoritmo generico per il proble del commesso viaggiatore
    :param graph: dict grafo
    :param generations: int generazioni
    :param chromosomes_number: int dimensione della popolazione
    :param selection_size: int indice di selezione
    :param roulette: boolean modalita di selezione
    :param mutation_rate: int indice di mutazione
    :param vertex_number: int numero di stazioni
    :return:
    """
    print(graph)
    fitness_func = get_fitness_func(graph, vertex_number)
    population = get_first_generations(int(chromosomes_number), vertex_number)
    best_chromo = population[0]
    best_fit = 0
    for generation in range(0, generations):
        print('Generation {}'.format(generation))
        try:
            best_fit = fitness_func(best_chromo)
            crossed = []
            if not roulette:
                selected = selection(population, selection_size, fitness_func)
                crossed = crossover(selected, crossover_func_pm, vertex_number, chromosomes_number)
                crossed = selection(selected + crossed, chromosomes_number, fitness_func)
            else:
                crossed = roulette_crossover(population, fitness_func, crossover_func_pm, vertex_number,
                                             chromosomes_number)
                crossed = selection(crossed, chromosomes_number, fitness_func)
            mutated = mutation(crossed, vertex_number, mutation_rate)
            population = mutated
            for chromosome in population:
                fit = fitness_func(chromosome)
                if fit < best_fit:
                    best_chromo = chromosome
            print('Best chromo of generation {} '.format(generation), best_fit, best_chromo)
        except Exception as e:
            print(e)
            raise
    return best_fit, best_chromo


@click.command()
@click.option('--generations', default=1, help='Number of generations.')
@click.option('--init-size', prompt='Give starting chromosomes group size',
              help='Size of starting chromosome group.')
@click.option('--selection-size', prompt='Give selection size',
              help='The selection size.')
@click.option('--roulette', prompt='Roulette Wheel',
              help='Use roulette wheel method for crossover', type=bool)
@click.option('--mutation-rate', prompt='Give mutation rate',
              help='The mutation rate.')
@click.option('--input-file', prompt='Give input file',
              help='The data file.')
@click.option('--test-repetitions', prompt='Give execution repetition number',
              help='How may time to repeat execution.')
def salesman(generations, init_size, selection_size, roulette, mutation_rate, input_file, test_repetitions):
    """
    Test dell applicazione di un algoritmo genetico al problema del comesso viaggiatore
    :param generations:
    :param init_size:
    :param selection_size:
    :param mutation_rate:
    :param input_file:
    :param test_repetitions:
    :return:
    """
    start_time = time.time()
    print('<----------- Starting {} Executions ------------->'.format(test_repetitions))
    print('<----------- Code Version {} ------------->'.format(VERSION))

    name, description, graph, vertex_number = parse_xml(input_file)
    print('File name: {}'.format(name))
    print('Data descriptions: {}'.format(description))
    print('Number of generations: {}'.format(generations))

    test_results = []
    chromosomes = []

    for index in range(1, int(test_repetitions) + 1):
        best_fit, best_chromo = ga(graph, int(generations), int(init_size), int(selection_size), roulette,
                                   int(mutation_rate),
                                   vertex_number)
        test_results.append(best_fit)
        chromosomes.append(best_chromo)
    print("<-------- Executions time: {} -------->".format((time.time() - start_time)))

    best_execution = min(test_results)
    print('Best execution score {}, path {}'.format(best_execution, chromosomes[test_results.index(best_execution)]))
    worst_execution = max(test_results)
    print('Worst execution score {}, path {}'.format(worst_execution, chromosomes[test_results.index(worst_execution)]))
    average = np.average(test_results)
    print('Average execution score {}'.format(average))

    plt.plot(test_results, '-bo')
    plt.hlines(average, xmin=0, xmax=int(test_repetitions) - 1, colors='#E8DD22', linestyles='dashed', label='Average')
    plt.hlines(worst_execution, xmin=0, xmax=int(test_repetitions) - 1, colors='tab:red', linestyles='dashed',
               label='Worst')
    plt.hlines(best_execution, xmin=0, xmax=int(test_repetitions) - 1, colors='tab:green', linestyles='dashed',
               label='Best')
    plt.ylabel('Results')
    plt.show()


if __name__ == '__main__':
    salesman()
