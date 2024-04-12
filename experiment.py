import networkx as nx
import random
import utils
import tqdm
import nest

file = "cora.gml"

training_size = 20

G = nx.Graph(nx.read_gml(file))

# seperate and set up training and test neurons
class_table = {n: c for n, c in nx.get_node_attributes(G, "class").items()}
classes = set(class_table.values())

# For each of the three citation network graphs, we randomly select 20 papers per topic to serve as our training set.
training_nodes = {k: set() for k in classes}
all_train = set()
rand_nodes = list(G.nodes())
random.shuffle(rand_nodes)
for n in rand_nodes:
    c = class_table[n]
    if len(training_nodes[c]) < training_size:
        training_nodes[c].add(n)
        all_train.add(n)

subset = 100

l = list(class_table.items())
random.shuffle(l)
l = l[:subset]

total = 0
count = 0
for n, c in tqdm.tqdm(l):
    if n not in all_train:
        citation_neurons, class_neurons, recorders = utils.setup_nest(G, all_train)

        # NOTE: you need a high time step count for some reason
        pred_c = utils.run_simulation(n, citation_neurons, class_neurons, reset=False, time_steps=20000)

        outs = 0
        for l, sr in recorders.items():
            outs += sr.get(["n_events"])["n_events"]

        nest.ResetKernel()

        print(f"predicted: {pred_c}, actual: {c}")
        total += 1 if pred_c == c else 0
        count += 1

print(f"Accuracy: {total / count}")
