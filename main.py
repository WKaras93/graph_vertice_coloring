import graph

g = graph.Graph('myciel3.txt')
params = {'populationSize':50, 'crossoverPropability':0.95, 'mutationPropability': 0.4, 'limitIteration':20000, 'crossoverMethod':'one-point', 'expectedColors': 4}
ga = graph.GeneticAlgorithm(g, params)
print(ga.assignColorToVertice())
sl = graph.SmallestLastAlgorithm(g)
print(sl.assign_color_to_vertices())