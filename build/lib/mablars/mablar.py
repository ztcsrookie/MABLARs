import sys
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
from .membership_functions import *
from .dataloaders import *
from sklearn.preprocessing import LabelEncoder

from weka.core.converters import Loader, load_any_file
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.classifiers import Evaluation
from weka.core.dataset import create_instances_from_matrices, create_instances_from_lists, Instance


class mb:
    def __init__(self, causal_matrix=None, causal_dag=None, mb_list=[], rule_base=None, linguictic_rule_base=None,
                 fuzzy_sets_parameters=None, node_name_list=None, cg_save_path='test.png'):
        '''
        an object using MABLAR.
        :param causal_matrix: a D*D numpy matrix, where D is the number of variables. causal_matrix[i,j]=1 represents that i is the cause of j.
        :param causal_dag: a dag object of the causal-learn package.
        :param mb_list: a list indicates the Markov blanket of the target variable.
        :param rule_base: an R*D numpy matrix if using the WM algorithm for rule generation, where R is the number of rules and D is the number of variables.
        :param linguictic_rule_base: The linguistic rule base
        :param fuzzy_sets_parameters: The parameters of fuzzy sets of each variable.
        :param node_name_list: The name of each variable.
        :param cg_save_path: The save path to save the obtained causal graph.
        '''
        self.causal_matrix = causal_matrix
        self.causal_dag = causal_dag
        self.mb_list=mb_list
        self.rule_base = rule_base
        self.linguistic_rule_base = linguictic_rule_base
        self.fuzzy_sets_parameters = fuzzy_sets_parameters
        self.node_name_list = node_name_list
        self.cg_save_path = cg_save_path

    def fit_pc_anm(self, le_data):
        '''
        Causal discovery using the PC algorithm and the ANM algorithm
        :param le_data: label_encoded data, i.e., string labels have already been encoded. All values should be numeric values.
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

    def fit_ica(self, le_data, current_seeds=3):
        '''
        Causal discovery using ICA-LiNGAM algorithm
        :param le_data: label_encoded data, i.e., string labels have already been encoded. All values should be numeric values.
        :param current_seeds: random seed for ICA
        :return:
        '''
        cd_model = lingam.ICALiNGAM(random_state=current_seeds, max_iter=10000)
        cd_model.fit(le_data)
        weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
        self.causal_matrix = weighted_causal_matrix.T
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
        '''
        Find the index of the Markov blanket of the target variable.
        :param causal_matrix: The causal matrix of the given data set
        :param x: the input data set
        :return:
        '''
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
        self.mb_list = list(markov_blanket)

        # return list(markov_blanket)

    def show_causal_graph(self, causal_dag):
        """
        Show the causal graph and save to the save path.
        :param causal_dag: The direct causal graph, dag of the causal-learn package.
        :param save_path: The save path.
        :return:
        """
        save_path = self.cg_save_path
        graphviz_dag = GraphUtils.to_pgv(causal_dag)
        graphviz_dag.draw(save_path, prog='dot', format='png')
        img = mpimg.imread(save_path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def rule_generation_wm(self, x, y, n_clusters=3) -> None:
        '''
        Generate the rule base using the WM algorithm
        :param x: the features, nparray N*D, N: #samples, D: #features. All values should be numeric values.
        :param y: the label. nparray N*1. N: #samples,
        :return:
        '''
        N, D = x.shape

        # u, cntr, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        #     x, n_clusters, 1.5, error=0.00001, maxiter=10000, init=None)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', max_iter=1000, tol=1e-6, algorithm='elkan')
        # kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=20, max_iter=1000, tol=1e-6, algorithm='elkan')
        kmeans.fit(x)
        cntr = kmeans.cluster_centers_

        self.fuzzy_sets_parameters = np.sort(cntr, axis=0)
        rule_base_dict = {}
        linguistic_rule_base = {}
        md_dict = {}

        for idx, sample in enumerate(x):
            current_product_md = 1
            candidate_rule = ()
            candidate_linguistic_rule = ()
            for i, feature_value in enumerate(sample):
                a = self.fuzzy_sets_parameters[0, i]
                b = self.fuzzy_sets_parameters[1, i]
                c = self.fuzzy_sets_parameters[2, i]
                md_low = left_shoulder_mf(feature_value, a, b)
                md_medium = triangle_mf(feature_value, a, b, c)
                md_high = right_shoulder_mf(feature_value, b, c)
                memberships = [
                    (md_low, 0, "Low"),
                    (md_medium, 1, "Mid"),
                    (md_high, 2, "High")
                ]
                max_md_i = max([md[0] for md in memberships])
                current_product_md *= max_md_i
                antecedent = max(memberships, key=lambda item: item[0])[1]
                linguistic_antecedent = max(memberships, key=lambda item: item[0])[2]
                candidate_rule += (antecedent,)
                candidate_linguistic_rule += (linguistic_antecedent,)

            if candidate_linguistic_rule not in linguistic_rule_base:
                linguistic_rule_base[candidate_linguistic_rule] = y[idx]
                rule_base_dict[candidate_rule] = y[idx]
                md_dict[candidate_linguistic_rule] = current_product_md

            elif candidate_linguistic_rule in linguistic_rule_base and current_product_md > md_dict[
                candidate_linguistic_rule]:
                linguistic_rule_base[candidate_linguistic_rule] = y[idx]
                rule_base_dict[candidate_rule] = y[idx]
                md_dict[candidate_linguistic_rule] = current_product_md

        numeric_rule_base = np.zeros((len(rule_base_dict), D + 1))
        r = 0
        for key, value in rule_base_dict.items():
            for i, idx in enumerate(key):
                numeric_rule_base[r, i] = idx
            numeric_rule_base[r, -1] = value
            r += 1
        self.rule_base = numeric_rule_base
        self.linguistic_rule_base = linguistic_rule_base


    def furia_rule_generation(self, x, y):
        '''
        Generate the rule base using the FURIA algorithm
        :param x: the input of the data set. All values should be numeric values.
        :param y: the label (or output) of the data set.
        :return:
        '''
        furia = Classifier(classname="weka.classifiers.rules.FURIA")
        furia.build_classifier(x)
        return furia

    def fit(self, data, cd='pc_anm', rg='wm'):
        '''
        Train a fuzzy system using MABLAR
        :param data:the input data, N*D, where N is #samples, D is #variables. All values should be numeric values.
        The last column of data is the target variable. The label of the target variable should be integer, not string.
        :param cd: Select the causal discovery (cd) method. Options: 'pc_anm' and 'ica', default 'pc_anm'.
        :param rg: Select the rule generation method. 'wm': using the WM algorithm. 'furia': using the FURIA algorithm. Default 'wm'.
        :return:
        '''
        if cd == 'pc_anm':
            self.fit_pc_anm(data)
        if cd == 'ica':
            self.fit_ica(data)

        x = data[:, :-1]
        y = data[:, -1]
        self.find_mb(self.causal_matrix)
        self.mb_list

        if not self.mb_list:
            print('No mb')
            sys.exit(0)  # 成功退出
        else:
            mb_x = x[:, self.mb_list]

        if rg == 'wm':
            self.rule_generation_wm(mb_x, y)
        if rg == 'furia':
            jvm.start(packages=True)
            train_features_list = [self.node_name_list[i] for i in self.mb_list]
            train_features_list.append(self.node_name_list[-1])
            # arff_data = create_instances_from_matrices(mb_x, y, name="generated from matrices",
            #                                            cols_x=train_features_list[:-1], col_y=train_features_list[-1])
            arff_data = create_instances_from_matrices(mb_x, y, name="generated from matrices")
            arff_data.class_is_last()
            if arff_data.class_attribute.is_numeric:
                numeric_to_nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal",
                                            options=["-R", "last"])
                numeric_to_nominal.inputformat(arff_data)
                arff_data = numeric_to_nominal.filter(arff_data)
            furia = Classifier(classname="weka.classifiers.rules.FURIA")
            furia.build_classifier(arff_data)
            self.rule_base = furia
            self.linguistic_rule_base = furia

    def predict_wm(self, x) -> int:
        '''
        predict the label of the given sample, i.e., x, using the scfs_strategy.
        :param x: an input sample. All values should be numeric values.
        :return: y: the predicted label
        '''
        n_rules = self.rule_base.shape[0]  # number of rules
        rule_labels = self.rule_base[:, -1]
        D = self.rule_base.shape[1] - 1  # number of inputs
        md_rules = np.ones((n_rules, D))
        for r in range(n_rules):
            for d in range(D):
                # print(x[d])
                a = self.fuzzy_sets_parameters[0, d]
                b = self.fuzzy_sets_parameters[1, d]
                c = self.fuzzy_sets_parameters[2, d]
                if self.rule_base[r, d] == 0:
                    md_rules[r, d] = left_shoulder_mf(x[d], a, b)
                elif self.rule_base[r, d] == 1:
                    md_rules[r, d] = triangle_mf(x[d], a, b, c)
                else:
                    md_rules[r, d] = right_shoulder_mf(x[d], b, c)
        prod_md_rules = np.prod(md_rules, axis=1, keepdims=True)
        rule_labels = rule_labels.reshape(-1, 1)
        label_firing_strength = np.concatenate((prod_md_rules, rule_labels), axis=1)
        unique_labels = np.unique(rule_labels)
        max_firing_strength = 0
        for label in unique_labels:
            copy_label_firing_strength = copy.deepcopy(label_firing_strength)
            current_label_firing_strength = copy_label_firing_strength[copy_label_firing_strength[:, -1] == label]
            current_sum_md = np.sum(current_label_firing_strength[:, 0])
            if current_sum_md >= max_firing_strength:
                predict_y = label
                max_firing_strength = current_sum_md

        # max_row_index = np.argmax(prod_md_rules)
        # predict_y = self.rule_base[max_row_index, -1]

        return predict_y

    def predict_furia(self, x):
        '''

        :param x: the given sample. All values should be numeric values.
        :return: y: the predicted label.
        '''
        # x_list = x.tolist()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        jvm.start()
        weka_instance = Instance()
        input_x = weka_instance.create_instance(x,classname='weka.core.DenseInstance')
        # input_x = create_instances_from_matrices(x, name="generated from matrix (no y)")
        print(input_x)
        furia = self.rule_base
        predict_y = furia.classify_instance(input_x)
        jvm.stop()

        return predict_y


    # def predict_wm_winner_takes_all(self, x) -> int:
    #     '''
    #     winner_takes_all
    #     :param x: an input sample
    #     :return: y: the predicted label
    #     '''
    #     n_rules = self.rule_base.shape[0]
    #     D = self.rule_base.shape[1] - 1
    #     md_rules = np.ones((n_rules, D))
    #     for r in range(n_rules):
    #         for d in range(D):
    #             # print(x[d])
    #             a = self.fuzzy_sets_parameters[0, d]
    #             b = self.fuzzy_sets_parameters[1, d]
    #             c = self.fuzzy_sets_parameters[2, d]
    #             if self.rule_base[r, d] == 0:
    #                 md_rules[r, d] = left_shoulder_mf(x[d], a, b)
    #             elif self.rule_base[r, d] == 1:
    #                 md_rules[r, d] = triangle_mf(x[d], a, b, c)
    #             else:
    #                 md_rules[r, d] = right_shoulder_mf(x[d], b, c)
    #     prod_md_rules = np.prod(md_rules, axis=1, keepdims=True)
    #     max_row_index = np.argmax(prod_md_rules)
    #
    #     predict_y = self.rule_base[max_row_index, -1]
    #     return predict_y