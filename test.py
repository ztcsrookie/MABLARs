from mablar_cd import *

if __name__ == '__main__':
    data_path = '../CSV_Datasets/Breast.csv'
    data_loader = DataLoader()
    data, node_name_list = data_loader.load_csv_data(data_path)
    model = mbcd(node_name_list=node_name_list)
    model.fit(data, cd='pc_anm', rg='furia')
    model.show_causal_graph(model.causal_dag)
    print(model.linguistic_rule_base)
