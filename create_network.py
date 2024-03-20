import networkx as nx
import nest
import random

def setup_nest(gml_file, training_set):
    G = nx.Graph(nx.read_gml(gml_file))

    class_table = nx.get_node_attributes(G, "class")
    classes = set(class_table.values())

    citation_neurons = nest.Create("iaf_psc_alpha", len(G), {"tau_m": 100000, "V_reset": 0, "V_th": 1})
    class_neurons = nest.Create("iaf_psc_alpha", len(classes), {"tau_m": 100000, "V_reset": 0, "V_th": 1, "tau_minus": 30})

    nest.SetStatus(citation_neurons, {"label": [n for n in G.nodes()]})
    nest.SetStatus(class_neurons, {"label": [i for i in range(len(classes))]})

    # seperate and set up training and test neurons
    training_set = set(training_set)

    for n in G.nodes():
        if n not in training_set:
            citation_neurons[n].set({"t_ref": 100000, "tau_minus": 30})

    for u, v in G.edges():
        nest.Connect(citation_neurons[u], citation_neurons[v], syn_spec={"weight": 100, "delay": 1})
        nest.Connect(citation_neurons[v], citation_neurons[u], syn_spec={"weight": 100, "delay": 1})

    for n, c in nx.get_node_attributes(G, "class").items():
        assert nest.GetStatus(class_neurons[c], "label") == c

        if n in training_set:
            nest.Connect(citation_neurons[n], class_neurons[c], syn_spec={"weight": 1, "delay": 1, "label": c})
            nest.Connect(class_neurons[c], citation_neurons[n], syn_spec={"weight": 1, "delay": 1, "label": c})
        else:
            nest.Connect(citation_neurons[n], class_neurons[c], syn_spec={"synapse_model": "stdp_synapse", "weight": 0.0001, "delay": 1, "label": c})
            nest.Connect(class_neurons[c], citation_neurons[n], syn_spec={"synapse_model": "stdp_synapse", "weight": 0.0001, "delay": 1, "label": c})

    return citation_neurons, class_neurons

def run_simulation(test_node_id, citation_neurons, time_steps=20, reset=False):
    test_neuron = citation_neurons[test_node_id]
    spike = nest.Create("spike_generator", params={"spike_times": [1.0]})
    nest.Connect(spike, test_neuron)

    # default nest values
    tics_per_step = 100.0
    ms_per_tic = 0.001

    nest.Simulate(time_steps * tics_per_step * ms_per_tic)

    # getting the highest node --> class synapse is identical to the lowest class --> node synapse
    weights = nest.GetConnections(source=test_neuron, synapse_model="stdp_synapse").get(["weight", "label"])

    highest = max(weights.values(), key=lambda x: x[0])[1]

    if reset:
        nest.ResetKernel()

    return highest
