"""
Batch a list of AAG-CNF graphs
"""
import torch
import dgl
import networkx
from dgl_graph_base import DGL_Graph_Base
import time
    
def batched_combined_graph(L):
    """L is a list of at least 1 DGL Heterograph"""
    l2c_indices_0 = []
    l2c_indices_1 = []
    aagf_indices_0 = []
    aagf_indices_1 = []
    lit_labels = []
    clause_labels = []
    num_lits, shift_lits = 0, 0
    num_clauses, shift_clauses = 0, 0
    
    for j, G in enumerate(L):
        shift_lits = num_lits
        shift_clauses = num_clauses
        num_lits += G.number_of_nodes('literal')
        num_clauses += G.number_of_nodes('clause')
        
        curr_l2c = G.edges(etype='l2c') 
        curr_aagf = G.edges(etype='aag_forward')
        l2c_indices_0.append(curr_l2c[0] + shift_lits)
        l2c_indices_1.append(curr_l2c[1] + shift_clauses)
        aagf_indices_0.append(curr_aagf[0] + shift_lits)
        aagf_indices_1.append(curr_aagf[1] + shift_lits)
        
        lit_labels.append(G.nodes['literal'].data['lit_labels'])
        clause_labels.append(G.nodes['clause'].data['clause_labels'])

    l2c_indices_0 = torch.cat(l2c_indices_0)
    l2c_indices_1 = torch.cat(l2c_indices_1)
    aagf_indices_0 = torch.cat(aagf_indices_0)
    aagf_indices_1 = torch.cat(aagf_indices_1)
    lit_labels = torch.cat(lit_labels, dim=0)
    clause_labels = torch.cat(clause_labels, dim=0)

    G = dgl.heterograph(
                {('literal', 'aag_forward', 'literal') : (aagf_indices_0, aagf_indices_1),
                 ('literal', 'aag_backward', 'literal') : (aagf_indices_1, aagf_indices_0),
                 ('literal', 'l2c', 'clause') : (l2c_indices_0, l2c_indices_1),
                 ('clause', 'c2l', 'literal') : (l2c_indices_1, l2c_indices_0)},
                {'literal': num_lits,
                 'clause': num_clauses}
    )
    G.nodes['literal'].data['lit_labels'] = lit_labels
    G.nodes['clause'].data['clause_labels'] = clause_labels
    return G

###############################################################################
    ### TEST
###############################################################################
#a = DGL_Graph_Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/a.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/a.qaiger.qdimacs')
#b = DGL_Graph_Base()
#b.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/b.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/b.qaiger.qdimacs')
#c = DGL_Graph_Base()
#c.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/c.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/c.qaiger.qdimacs')
#A, B, C = a.G, b.G, c.G
#G = batched_combined_graph2([A, B, C])
##A, B = a.G, b.G
##G = batched_combined_graph([A, B])
#print("*** A:")
#print(A['l2c'].adjacency_matrix())
#print("*** B:")
#print(B['l2c'].adjacency_matrix())
#print("*** C:")
#print(C['l2c'].adjacency_matrix())
#print("*** G:")
#print(G['l2c'].adjacency_matrix())
###############################################################################
#t1 = time.time()
#H = batched_combined_graph2([x.G for x in collated_batch.state.ext_data])
#print('batching 2 took {} seconds'.format(time.time()-t1))