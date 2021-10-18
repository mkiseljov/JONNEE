# diffusion embeddings generation

from collections import Counter
import networkx as nx
import numpy as np
import pandas as pd
import random
import time
from gensim.models import Word2Vec
from gensim.models.word2vec import logger, FAST_VERSION
import logging
import numpy.distutils.system_info as sysinfo
import scipy

from DuoGAE.utils import SequenceLearner

###############################################
###         Diff2Vec Implementation         ###
###############################################


class Diff2Vec(SequenceLearner):
    """
    Diff2Vec
    """
    def __init__(self, **kwargs):
        """
        args = {'vertex_set_cardinality': 80, 'num_diffusions': 10,
            'workers': 1, 'tb_type': 'eulerian', 'dimensions': 16, 
            'alpha': 0.025, 'window_size': 10, 'iter': 1, 'seed': 0}
        """
        self.name = 'diff2vec'

        self.vertex_set_cardinality = 80
        self.num_diffusions = 10
        self.tb_type = 'eulerian'
        self.alpha = 0.025
        self.window_size = 10
        self.iter = 3
        self.dimensions = 16
        self.seed = 0
        self.weighted = False

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])


    def learn_embeddings(self, nx_G):
        """
        :param dataset: Dataset instance
        """
        # self.n = dataset.n
        # nx_G = dataset.get_nx_graph()
        self.n = nx_G.number_of_nodes()
        sub_graphs = SubGraphComponents(nx_G, self.seed, self.weighted)
        sub_graphs.separate_subcomponents()
        sub_graphs.single_feature_generation_run(self.vertex_set_cardinality, self.tb_type)
        paths = sub_graphs.get_path_descriptions()
        self.embeddings = self._learn_from_sequences(paths)


###############################################
###           Subgraph Components           ###
###############################################


class SubGraphComponents:
    def __init__(self, nx_G, seed, weighted=False):
        self.seed = seed
        self.start_time = time.time()
        self.graph = nx_G
        # self.graph = nx.from_edgelist(pd.read_csv(edge_list_path, index_col = None).values.tolist())
        self.weighted = weighted
 
    def separate_subcomponents(self):
        # hoping that it preserves weights to use weighted_diffusion
        self.graph = sorted(nx.connected_component_subgraphs(self.graph),
                            key=len, reverse=True)

    def print_graph_generation_statistics(self):       
        print("The graph generation at run " + str(self.seed) + " took: " + str(round(time.time() - self.start_time, 3)) + " seconds.\n") 
        
    def single_feature_generation_run(self, vertex_set_cardinality, traceback_type):
        random.seed(self.seed)

        self.start_time = time.time()

        self.paths = {}

        for sub_graph in self.graph:
            nodes = list(sub_graph.nodes())
            random.shuffle(nodes)
            
            current_cardinality = len(nodes)
            
            if current_cardinality < vertex_set_cardinality:
                vertex_set_cardinality = current_cardinality
            for node in nodes:
                tree = EulerianDiffusionTree(node)
                # ??? condition on traceback type here ???
                if self.weighted:
                    tree.run_weighted_diffusion_process(sub_graph, vertex_set_cardinality)
                else:
                    tree.run_diffusion_process(sub_graph, vertex_set_cardinality)

                path_description = tree.create_path_description(sub_graph)
                self.paths[node] = list(map(lambda x: str(x), path_description))
                
        self.paths = self.paths.values()
                
    def print_path_generation_statistics(self):
        print("The sequence generation took: " + str(time.time() - self.start_time))
        print("Average sequence length is: " + str(np.mean(list(map(lambda x: len(x), self.paths)))))
        
    def get_path_descriptions(self):
        return self.paths



###############################################
###             Diffusion Trees             ###
###############################################


class EulerianDiffusionTree:
    
    def __init__(self, node):

        """
        Initializing a diffusion tree.

        :param node: Source of diffusion.
        """
        
        self.start_node = node
        self.infected = [node]
        self.infected_set = set(self.infected)
        self.sub_graph = nx.DiGraph()
        self.sub_graph.add_node(node)
        self.infected_counter = 1

    def run_diffusion_process(self, graph, number_of_nodes):

        """
        Creating a diffusion tree from the start node on G with a given vertex set size.
        The tree itself is stored in an NX graph object.

        :param graph: Original graph of interest.
        :param number_of_nodes: Cardinality of vertex set.
        """
        
        while self.infected_counter < number_of_nodes:
            
            end_point = random.sample(self.infected, 1)[0]
            sample = random.sample(list(graph.neighbors(end_point)), 1)[0]

            if sample not in self.infected:
                self.infected_counter = self.infected_counter + 1
                self.infected_set = self.infected_set.union([sample])
                self.infected = self.infected + [sample]
                self.sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])
                
                if self.infected_counter == number_of_nodes:
                    break


    def run_weighted_diffusion_process(self, graph, number_of_nodes):

        """
        Creating a diffusion tree from the start node on G with a given vertex set size.
        The tree itself is stored in an NX graph object.

        :param graph: Original graph of interest.
        :param number_of_nodes: Cardinality of vertex set.
        """

        while self.infected_counter < number_of_nodes:
            end_point = random.sample(self.infected, 1)[0]

            # sample these neighbors with weights
            neighbors = list(graph.neighbors(end_point))
            weights = [graph[end_point][nbhr]['weight']
                        for nbhr in neighbors]
            weights /= np.sum(weights)
            
            sample = np.random.choice(a=neighbors, p=weights, size=1)[0]

            if sample not in self.infected:
                self.infected_counter = self.infected_counter + 1
                self.infected_set = self.infected_set.union([sample])
                self.infected = self.infected + [sample]
                self.sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])
                
                if self.infected_counter == number_of_nodes:
                    break


    def create_path_description(self, graph):

        """
        Creating a random Eulerian walk on the diffusion tree.

        :param graph: Original graph of interest.
        """
        
        self.euler = [u for u,v in nx.eulerian_circuit(self.sub_graph, self.start_node)]
        if len(self.euler) == 0 :
            self.euler = [u for u,v in nx.eulerian_circuit(graph, self.start_node)]
        return self.euler



####### Another type of diffusion tree ####### 

class EndPointDiffusionTree:

    def __init__(self, node):

        """
        Initializing a diffusion tree.

        :param node: Source of diffusion.
        """

        self.needed_nodes = [node]
        self.diffusion_set =  [self.needed_nodes]
        self.start_node = node
        self.infected = set(self.needed_nodes)
        
    def run_diffusion_process(self, graph, number_of_nodes):

        """
        Creating a diffusion tree from the start node on G with a given vertex set size.
        The tree itself is stored in a list of lists.

        :param graph: Original graph of interest.
        :param number_of_nodes: Cardinality of vertex set.
        """

        while len(self.infected) < number_of_nodes:
            diffusion_set_to_be_added = []
            end_point = self.diffusion_set[0][-1]
            sample = random.sample(graph.neighbors(end_point), 1)[0]
            if sample not in self.infected:
                diffusion_set_to_be_added = diffusion_set_to_be_added + [self.diffusion_set[0] + [sample]]
                self.infected = self.infected.union([sample])
                self.needed_nodes = self.needed_nodes + [end_point, sample]
                if len(self.infected) == number_of_nodes:
                    break
                self.diffusion_set = self.diffusion_set + diffusion_set_to_be_added
            random.shuffle(self.diffusion_set)

    def transform_infected_nodes(self):

        """
        Creating a set of nodes with degree equal to 1.  
        """

        self.needed_nodes = Counter(self.needed_nodes)
        self.needed_nodes = [needed_node for needed_node in self.needed_nodes if self.needed_nodes[needed_node] == 1]
        self.infected = set(self.needed_nodes)
        
    
    def filter_path_chunks(self):

        """
        Filtering the diffusion paths that end in a vertex with degree equal to 1.
        """

        self.diffusion_set = filter(lambda x: (x[0] == self.start_node) and (x[-1] in self.infected) and len(x) > 1, self.diffusion_set)
    
    def create_path_description(self):

        """
        Creating an endpoint traceback on the diffusion tree.
        """      

        self.transform_infected_nodes()

        if len(self.diffusion_set) > 1:

            self.paths = []
            self.filter_path_chunks()

            for path in self.diffusion_set:

                reverse_path = path[::-1]
                out_path = path[1::] + reverse_path[1:len(path)-1]

                if len(out_path) > 1:

                    self.paths = self.paths + out_path
        else:

            self.paths = self.diffusion_set
    
        self.paths = [self.start_node] + self.paths

        return self.paths


