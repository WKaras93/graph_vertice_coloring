import random
import re
from copy import deepcopy


class Graph:
    """The Graph class is the main structure that stores the graph as a adjacency list. 
    The class supports elementary methods for graphs."""
    def __init__(self, dictionary):
        """The method takes one parameter: path to file or dictionary contains graph."""
        self.adjacency_list = {}
        if isinstance(dictionary, dict):
            self.adjacency_list = dictionary
        elif isinstance(dictionary, str):
            self._load_from_file(dictionary)
        else:
            raise TypeError('Wrong type of insert data')

    def _load_from_file(self, file_name):
        """
        This method supports reading graphs from file. File structrue for import the dictionary:
            v: u, x 
            where exist edge (v, u) and (v, x).
        File structure for import edges:
            e v u
        where exist edge e means edge and u and v are vertices of edge.
        """
        with open(file_name, 'r') as file:
            for line in file:
                line = (re.sub('[:,\n]', ' ', line)).split()
                if line[0] == 'e':
                    self._add_edge(int(line[1]), int(line[2]))
                else:
                    self.adjacency_list[line[0]] = line[1:]
        file.close()
    
    def _add_edge(self, v, u):
        if v in self.adjacency_list:
            self.adjacency_list[v].append(u)
        else:
            self.adjacency_list[v] = [u]
        if u in self.adjacency_list:
            self.adjacency_list[u].append(v)
        else:
            self.adjacency_list[u] = [v]

    def _get_vertices(self):
        """Returns all vertices from graph as a list."""
        return [v for v in self.adjacency_list.keys()]

    def _get_edges(self):
        """Returns all edges from graph as a list of pairs."""
        return [(u, v) for u in self.adjacency_list.keys() for v in self.adjacency_list[u]]

    def _get_neighbours(self, vertice):
        """Returns all neighbours passed in as a vertex parameter as a list."""
        return self.adjacency_list[vertice]
    
    def _is_empty(self):
        """Checks if the graph is empty."""
        return self.adjacency_list == {}

    def _get_vertices_degree(self):
        """Returns degree each vertice of the graph as a dictionary, where the key is the name of vertice
        and the value is degree."""
        vertices_degree = {}
        for vertice in self._get_vertices():
            vertices_degree[vertice] = len(self.adjacency_list[vertice])
        return vertices_degree
    
    def _remove_vertice(self, remove_vertice):
        """Removes the vertice passed as a parameter and all edges that contain it."""
        for vertice in self.adjacency_list.values():
            try:
                vertice.remove(remove_vertice)
            except ValueError:
                pass
        del self.adjacency_list[remove_vertice]


class GeneticAlgorithm:
    """GeneticAlgorithm is a class that supports the genetic algorithm for graph vertices
    coloring.""" 
    def __init__(self, graph, param):
        """
        The method retrives two parameters, it is a class graph object and param which is 
        a dictionary with the following keys:
            populationSize up number of vertices
            crossoverPropability from 0 to 1 (0.95)
            mutationPropability from 0 to 1 (0.1)
            limitIteration up 1
        The default values of the program are given in brackets above.
        """
        self.graph = graph
        self.edges = graph._get_edges()
        self.vertices = graph._get_vertices()
        self.population_size = param['populationSize'] if param['populationSize'] else len(self.graph._get_vertices())
        self.crossover_propability = param['crossoverPropability'] if param['crossoverPropability'] else 0.95
        self.mutation_propability = param['mutationPropability'] if param['mutationPropability'] else 0.1
        self.limit_iterations = param['limitIteration'] if param['limitIteration'] else 1

    def _get_number_of_edges(self):
        """Returns number of occurences of edges in the graph."""
        return len(self.edges)

    def _get_number_of_vertices(self):
        """Returns number of occurences of vertices in the graph."""
        return len(self.vertices)

    def assignColorToVertice(self, crossover_method = 'one-point', generate_statistics = False):
        """
        The main method supporting the genetic algorithm of vertex coloring. It takes two parameters that are optional.
        The first 'crossover_method' concerning method of crossing: 'one-point' / 'alternate'. The second paramater 
        'generate_statistics': False / True calls method writing to the file the best and worst chromosome each population
        and final best coloring.
        """
        population = self._initialize_population()
        iteration = 0
        result = []
        prev_fitness_result = 1
        current_fitness_result = 0
        while not self._stop_algorithm(iteration, [current_fitness_result, prev_fitness_result]):
            new_population = []
            while len(new_population) < len(population):
                parents = self._ranking_selection_method(population)
                if crossover_method == 'alternate':
                    children = self._alternate_crossover(parents)
                else:
                    children = self._crossover(parents)
                children = self._mutation(children)
                new_population.extend(children)
            population, best_fitness = self._select_new_population(population, new_population)
            result.append([population[0], best_fitness])
            if(len(result) >= 2):
                current_fitness_result = result[-1][1]
                prev_fitness_result = result[-2][1]
            iteration += 1
        print(iteration)
        return self._get_best_population(result)

    def _crossover(self, parents):
        """The method supporting one-point crossover, takes one parameter - a dictionary of candidates for parents.
        Returns dictionary with 2 elements."""
        parent_vertices = parents[0].keys()
        children = []
        while len(children) < 2:
            crossLine = random.randint(1, self._get_number_of_vertices()-2)
            crossover_propability = random.uniform(0, 1)
            if crossover_propability <= self.crossover_propability:
                first_parent_colors = list(random.choice(parents).values())
                second_parent_colors = list(random.choice(parents).values())
                child1 = dict(zip(parent_vertices, first_parent_colors[:crossLine] + second_parent_colors[crossLine:]))
                child2 = dict(zip(parent_vertices, second_parent_colors[:crossLine] + first_parent_colors[crossLine:]))
                if self._has_correct_color(child1):
                    children.append(child1)
                elif self._has_correct_color(child2):
                    children.append(child2)
            else:
                children.append(random.choice(parents))
        return children

    def _alternate_crossover(self, parents):
        """The method supporting alternate crossover, takes one parameter - a list of candidates for parents.
        Returns dictionary with 2 elements."""
        crossover_propability = random.uniform(0, 1)
        parent_vertices = list(parents[0].keys())
        children = []
        for i in range(2):
            if crossover_propability <= self.crossover_propability:
                child = {}
                first_parent_colors = list(random.choice(parents).values())
                second_parent_colors = list(random.choice(parents).values())
                for i in range(len(parent_vertices)):
                    if i % 2 == 0 and self._neighbours_have_different_color(parent_vertices[i], first_parent_colors[i], child):
                        child[parent_vertices[i]] = first_parent_colors[i]
                    elif self._neighbours_have_different_color(parent_vertices[i], second_parent_colors[i], child):
                        child[parent_vertices[i]] = second_parent_colors[i]
                    else:
                        color_palette = [i for i in range(
                            self._get_number_of_vertices())]
                        vertice_color = random.choice(color_palette)
                        while not self._neighbours_have_different_color(parent_vertices[i], vertice_color, child):
                            vertice_color = random.choice(color_palette)
                        child[parent_vertices[i]] = vertice_color
                children.append(child)
            else:
                children.append(random.choice(parents))
        return children

    def _neighbours_have_different_color(self, vertice_name, vertice_color, child):
        """The method checks if the neighbouring vertices of the vertice give as the parameter have a different color."""
        for neighbour in self.graph._get_neighbours(vertice_name):
            if neighbour in child and vertice_color == child[neighbour]:
                return False
        return True

    def _select_new_population(self, prev_population, current_population):
        """The method returns new population created on two previous ones."""
        fitness_result = self._fitness(prev_population)
        new_fitness_result = self._fitness(current_population)
        population_size = self.population_size
        sum_of_population = dict(zip([i for i in range(
            population_size * 2)], fitness_result + new_fitness_result))
        sum_of_population = sorted(sum_of_population.items(), key=lambda kv: (kv[1], kv[0]))
        fitness_for_best = sum_of_population[0][1]
        index = [n[0] for n in sum_of_population[:self.population_size]]
        new_population = []
        for i in index:
            if i < population_size:
                new_population.append(prev_population[i])
            else:
                new_population.append(current_population[i % population_size])
        return [new_population, fitness_for_best]

    def _ranking_selection_method(self, population):
        """The method returns the best individuals from population."""
        fitness_result = self._fitness(population)
        fitness_result = dict(zip([i for i in range(self.population_size)], fitness_result))
        sorted_fitness = sorted(fitness_result.items(), key=lambda kv: (kv[1], kv[0]))
        numbers = [n[0] for n in sorted_fitness[:int(self.population_size/2)]]
        return [population[n] for n in numbers]

    def _fitness(self, population):
        """The method calculates the adjustment of the individual in the population.""" 
        colors = [colors for colors in [len(set(genotype.values())) for genotype in population]]
        sum_colors = sum(colors)
        return [color * color * sum_colors for color in colors]

    def _mutation(self, children):
        """This method mutates passed on individuals."""
        mutation_propability = random.uniform(0, 1)
        if mutation_propability < self.mutation_propability:
            vertices_number = self._get_number_of_vertices()-1
            mutated_position = random.randint(0, vertices_number)
            color = random.randint(0, vertices_number)
            mutated_child_position = random.randint(0, 1)
            mutated_child_color = list(children[mutated_child_position].values())
            mutated_child_name = list(children[mutated_child_position].keys())
            mutated_child_color[mutated_position] = color
            mutated_child = dict(zip(mutated_child_name, mutated_child_color))
            if self._has_correct_color(mutated_child):
                children[mutated_child_position] = mutated_child
        return children

    def _get_best_population(self, children):
        """Returns the best individual from set."""
        children.sort(key = lambda x: x[1])
        return (children[0][0], len(set(children[0][0].values())))

    def _stop_algorithm(self, numberOfIteration, fitness_populations):
        """The method stops the algorithm when the condition is met."""
        difference = abs(fitness_populations[0] - fitness_populations[1])
        if (numberOfIteration > self.limit_iterations or difference < 0.01 * fitness_populations[0]):
            return True
        else:
            return False

    def _initialize_population(self):
        """This method initialize first population."""
        initialed_population = []
        while (len(initialed_population) < self.population_size):
            color_palette = [i for i in range(self._get_number_of_vertices())]
            colored_vertices = {}
            for i in range(self._get_number_of_vertices()):
                v = self.vertices[i]
                color = random.choice(color_palette)
                colored_vertices[v] = color
            if self._has_correct_color(colored_vertices):
                initialed_population.append(colored_vertices)
        return initialed_population

    def _has_correct_color(self, colored_vertices):
        """Checks whether there is any color conflict for the solution provided."""
        for edge in self.edges:
            if colored_vertices[edge[0]] == colored_vertices[edge[1]]:
                return False
        return True


class SmallestLastAlgorithm:
    """SmallestLastAlgorithm is a class that supports the sequence algorithm.""" 
    def __init__(self, graph):
        """The method retrives one parameters, it is a class graph object."""
        self.graph = graph
        self.graph_length = len(self.graph._get_vertices())
    
    def assign_color_to_vertices(self):
        """The main method which runs SL algorithm, returns 2 values: graph coloring and number of colors used."""
        graph_coloring = self._greedily_coloring(self._sort_vertices_by_degree())
        number_of_colors = len(set(graph_coloring.values()))
        return (graph_coloring, number_of_colors)

    def _sort_vertices_by_degree(self):
        """This method returns a stack in which the vertices with the smallest degrees are added first."""
        graph_copy = Graph(deepcopy(self.graph.adjacency_list))
        stack = []
        while not graph_copy._is_empty():
            min_vertice_degree = min(graph_copy._get_vertices_degree().items(), key=lambda x:x[1])[0]
            stack.append(min_vertice_degree)
            graph_copy._remove_vertice(min_vertice_degree)
        return stack
    
    def _greedily_coloring(self, stack):
        """This method assign each vertice from stack the smallest possible value color."""
        vertices_color = {}
        while stack != []:
            is_colored = [False] * self.graph_length
            coloring_vertice = stack.pop()
            neighbours = self.graph._get_neighbours(coloring_vertice)
            for neighbour in neighbours:
                if neighbour in vertices_color:
                    is_colored[vertices_color.get(neighbour)] = True
            color = 0
            while is_colored[color] == True:
                color += 1
            vertices_color[coloring_vertice] = color
        return vertices_color