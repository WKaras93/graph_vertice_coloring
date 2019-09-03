import graph
import SLalgorithm

g = graph.Graph({1:[2,4], 2:[1,3], 3:[2,6], 4:[1,5], 5:[4,8], 6:[3,7], 7:[6,8], 8:[5,7]})
#g = graph.Graph({1:[2,5,4], 2:[1,5], 3:[5,6], 4:[1,6], 5:[1,2,3], 6:[3,4]})
# #g = graph.Graph({1:[2,3], 2:[1,4], 3:[1,4], 4:[2,3]})
params = {'populationSize':100, 'crossoverPropability':0.95, 'mutationPropability': 0.1, 'limitIteration':100}
ga = graph.GeneticAlgorithm(g, params)
print(ga.assignColorToVertice())
#g.load_from_file('graph.txt')
#g = graph.Graph({1:[2,4], 2:[1,3], 3:[2,6], 4:[1,5], 5:[4,8], 6:[3,7], 7:[6,8], 8:[5,7]})

g = graph.Graph({1:[2,5,4], 2:[1,5], 3:[5,6], 4:[1,6], 5:[1,2,3], 6:[3,4]})
sl = graph.SmallestLastAlgorithm(g)
# print(sl.assign_color_to_vertices())
graph.GraphicResult(g, sl.assign_color_to_vertices())