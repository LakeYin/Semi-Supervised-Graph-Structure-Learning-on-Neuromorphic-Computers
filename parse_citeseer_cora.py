import pandas as pd
import networkx as nx
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("write_path")
args = parser.parse_args()

dir = args.dir
write_path = args.write_path

cites = list(glob.glob(dir + "/*.cites"))[0]
content = list(glob.glob(dir + "/*.content"))[0]

G = nx.read_edgelist(cites, create_using=nx.DiGraph, nodetype=str).reverse()
G.remove_edges_from(nx.selfloop_edges(G))

df = pd.read_csv(content, sep="\t", header=None, index_col=0).iloc[:, -1].squeeze()

class_table = {k: v for v, k in enumerate(set(df))}

node_table = {str(node): {"class": class_table[label]} for node, label in zip(df.index, df)}

nx.set_node_attributes(G, node_table)

nx.write_gml(G, write_path)

