import copy

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import weka.core.jvm as jvm

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.ANM.ANM import ANM
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz, chisq
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from weka.core.converters import Loader, load_any_file
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.classifiers import Evaluation
from weka.core.dataset import create_instances_from_matrices

class CausalGraph:
    def __init__(self, causal_matrix=None, causal_dag=None, node_name_list=None, png_save_path='test_png'):
        '''
        :param causal_matrix: If A[i,j] !=0, the causal relationship: i -> j
        :param causal_dag: The causal graph object of the causal-learn library
        :param node_name_list: The node name list
        '''
        self.causal_matrix = causal_matrix
        self.causal_dag = causal_dag
        self.node_name_list = node_name_list
        self.save_path = png_save_path

    def fit_ica(self, le_data, current_seeds=3):
        '''
        Causal discovery using ICA-LiNGAM algorithm
        :param le_data: label_encoded data, i.e., string labels have already been encoded
        :param current_seeds: random seed for ICA
        :return:
        '''
        cd_model = lingam.ICALiNGAM(random_state=current_seeds, max_iter=10000)
        cd_model.fit(le_data)
        weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
        self.causal_matrix = weighted_causal_matrix.T
        self.causal_dag = self.create_dag_from_matrix(self.causal_matrix, self.node_name_list)

    def fit_pc_anm(self, le_data):
        '''
        Causal discovery using the PC algorithm and the ANM algorithm
        :param le_data: label_encoded data, i.e., string labels have already been encoded
        :return:
        '''
        CG = pc(le_data, 0.05, fisherz, True, 1, 2)
        cur_graph_matrix = copy.deepcopy(CG.G.graph)
        D = cur_graph_matrix.shape[0]
        # Determine the direction of the undirected edge.
        for i in range(D):
            for j in range(i):
                if cur_graph_matrix[i, j] == 0:
                    continue
                elif cur_graph_matrix[i, j] == cur_graph_matrix[j, i]:
                    data_x = le_data[:, i].reshape(-1, 1)
                    data_y = le_data[:, j].reshape(-1, 1)
                    anm = ANM()
                    p_value_foward, p_value_backward = anm.cause_or_effect(data_x, data_y)
                    if p_value_foward > p_value_backward:  # i-->j
                        cur_graph_matrix[i, j] = -1
                        cur_graph_matrix[j, i] = 1
                    else:
                        cur_graph_matrix[i, j] = 1
                        cur_graph_matrix[j, i] = -1
        self.causal_matrix = self.cg_matrix_to_adjacent_matrix(cur_graph_matrix)
        self.causal_dag = self.create_dag_from_matrix(self.causal_matrix, self.node_name_list)

    def create_dag_from_matrix(self, causal_matrix, node_names_list):
        '''
        Create a direct acyclic graph (DAG) from a matrix.
        :param causal_matrix: a matrix which represents the causal relationships between variables.
        D*D, causal_matrix[i,j]=1 means i-->j
        :param node_names_list: The name of each node.
        :return: A DAG.
        '''
        nodes = []
        for name in node_names_list:
            node = GraphNode(name)
            nodes.append(node)
        dag = Dag(nodes)
        num_variables = causal_matrix.shape[0]
        for i in range(num_variables):
            for j in range(num_variables):
                if causal_matrix[i, j] != 0:
                    dag.add_directed_edge(nodes[i], nodes[j])
        return dag

    def cg_matrix_to_adjacent_matrix(self, A):
        """
        Convert causal graph matrix where A[i,j] == -1 and A[j,i] == 1, then i->j to
        a matrix B where B[i,j] == 1 then i->j, otherwise 0.
        :param A: The causal graph matrix
        :return: B
        """
        B = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] == -1 and A[j, i] == 1:
                    B[i, j] = 1
        return B

    def find_mb(self, causal_matrix, x=-1) -> list:
        # 获取矩阵的维度
        num_var = causal_matrix.shape[0]
        # 1. 获取X的所有父节点
        parents_of_X = [i for i in range(num_var) if causal_matrix[i, x] != 0]

        # 2. 获取X的所有子节点
        children_of_X = [i for i in range(num_var) if causal_matrix[x, i] != 0]

        # 3. 获取X的所有子节点的其他父节点
        spouses_of_X = []
        for child in children_of_X:
            spouses_of_X.extend([i for i in range(num_var) if causal_matrix[i, child] != 0 and i != x])

        # 合并所有的组件并去除重复的以及目标变量，即最后一个变量
        markov_blanket = set(parents_of_X + children_of_X + spouses_of_X)
        markov_blanket.discard(num_var - 1)

        return list(markov_blanket)

    def find_mbcd(self, causal_matrix, x=-1) -> list:
        num_var = causal_matrix.shape[0]
        mbcd = []
        for i in range(num_var):
            if causal_matrix[i, x] != 0:
                mbcd.append(i)
        if not mbcd:
            return []
        else:
            mbcd_set = set(mbcd)
            if num_var - 1 in mbcd_set:
                final_mbcd = list(mbcd_set.discard(num_var - 1))
            else:
                final_mbcd = list(mbcd_set)
            return final_mbcd

    def show_causal_graph(self, causal_dag):
        """
        Show the causal graph and save to the save path.
        :param causal_dag: The direct causal graph, dag of the causal-learn package.
        :param save_path: The save path.
        :return:
        """
        save_path = self.save_path
        graphviz_dag = GraphUtils.to_pgv(causal_dag)
        graphviz_dag.draw(save_path, prog='dot', format='png')
        img = mpimg.imread(save_path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
