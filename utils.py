import networkx as nx
import nest
import random

def setup_nest(G, training_set):
    class_table = nx.get_node_attributes(G, "class")
    classes = set(class_table.values())

    # We turn off leak for both sets of neurons by setting their τm (membrane time constants) value to 100,000 ms, and we set their reset voltages to 0 mV and their threshold voltages to 1 mV
    # NOTE: I_e not documented in paper, but most examples seem to use 376 as a default value, and it doesn't work without this
    citation_neurons = nest.Create("iaf_psc_delta", len(G), {"tau_m": 100000, "V_reset": 0, "V_th": 1, "I_e": 376})
    class_neurons = nest.Create("iaf_psc_delta", len(classes), {"tau_m": 100000, "V_reset": 0, "V_th": 1, "I_e": 376, "tau_minus": 30})

    citation_neurons = {label: neuron for label, neuron in zip(G.nodes(), citation_neurons)}
    class_neurons = {c: neuron for c, neuron in zip(range(len(classes)), class_neurons)}

    recorders = {}
    for l, n in citation_neurons.items():
        sr = nest.Create("spike_recorder", {"label": l})
        recorders[l] = sr
        nest.Connect(n, sr)

    training_set = set(training_set)

    # For the test neurons, we set their refractory periods to be very high
    for n in G.nodes():
        if n not in training_set:
            citation_neurons[n].set({"t_ref": 100000})

    # For each edge connecting nodes A and B in the citation network, we create two synapses: one from A to B and one from B to A. We set these synapses to have a weight of 100 and a delay of 1 ms.
    for u, v in G.edges():
        nest.Connect(citation_neurons[u], citation_neurons[v], syn_spec={"weight": 100, "delay": 1})
        nest.Connect(citation_neurons[v], citation_neurons[u], syn_spec={"weight": 100, "delay": 1})

    for n, c in nx.get_node_attributes(G, "class").items():
        # Since we know the topics of the papers in the training set, we create two synapses between paper N’s neuron and topic T’s neuron: one from N to T and one from T to N, each with a weight of 1 and delay of 1.
        if n in training_set:
            nest.Connect(citation_neurons[n], class_neurons[c], syn_spec={"weight": 1, "delay": 1})
            nest.Connect(class_neurons[c], citation_neurons[n], syn_spec={ "weight": 1, "delay": 1})
        # we create a synapse to and from each of the topics for each paper neuron in the testing set. Those synapses have a weight of 0.0001 and a delay of 1 ms. The synapses between papers and topics all have STDP turned on, with a τ− (time constant for the depression part of STDP window) value of 30 ms.
        else:
            for c_t in classes:
                nest.Connect(citation_neurons[n], class_neurons[c_t], syn_spec={"synapse_model": "stdp_synapse", "weight": 0.0001, "delay": 1})
                nest.Connect(class_neurons[c_t], citation_neurons[n], syn_spec={"synapse_model": "stdp_synapse", "weight": 0.0001, "delay": 1})

    return citation_neurons, class_neurons, recorders

def run_simulation(test_node_id, citation_neurons, class_neurons, time_steps=20, reset=False):
    test_neuron = citation_neurons[test_node_id]

    # we create a single spike on the corresponding testing neuron at simulation time 1
    spike = nest.Create("spike_generator", params={"spike_times": [1.0], "spike_weights": [1e3]})
    nest.Connect(spike, test_neuron)

    # default nest value
    ms_per_step = 0.1

    # We allow the network to simulate for 20 time steps.
    nest.Simulate(time_steps * ms_per_step)

    # Then, we look at the synapses between all of the topics and that test paper (from the topic to the test paper)
    # It is worth noting that we could similarly check the highest weighted synapse between the test paper and all of the topics.
    result_table = {}
    for c, class_neuron in class_neurons.items():
        weight = nest.GetConnections(source=class_neuron, target=test_neuron).get(["weight"])
        result_table[c] = weight["weight"]

    highest = min(result_table.items(), key=lambda x: x[1])[0]

    if reset:
        nest.ResetKernel()

    return highest
