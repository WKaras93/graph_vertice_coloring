import graph

#g = graph.Graph({1:[2,3], 2:[1,3], 3:[1,2]})
g = graph.Graph('edges.txt')
params = {'populationSize':10, 'crossoverPropability':0.95, 'mutationPropability': 0.1, 'limitIteration':100}
ga = graph.GeneticAlgorithm(g, params)
print(ga.assignColorToVertice('alternate'))
sl = graph.SmallestLastAlgorithm(g)
print(sl.assign_color_to_vertices())