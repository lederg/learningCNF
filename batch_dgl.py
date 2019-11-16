"""
Batch a list of AAG-CNF graphs
"""
import torch
import dgl
import networkx
from qbf_data import CombinedGraph1Base
import time


def batched_combined_graph(L):
    """L is a list of at least 1 DGL Heterograph"""
    l2c_indices_0 = []
    l2c_indices_1 = []
    aagf_indices_0 = []
    aagf_indices_1 = []
    
    lit_labels = torch.empty(size=[0, L[0].nodes['literal'].data['lit_labels'].shape[1]])
    clause_labels = torch.empty(size=[0, L[0].nodes['clause'].data['clause_labels'].shape[1]])
    
    num_lits, shift_lits = 0, 0
    num_clauses, shift_clauses = 0, 0
    
#    t2 = time.time()
    for j, G in enumerate(L):
        shift_lits = num_lits
        shift_clauses = num_clauses
        num_lits += G.number_of_nodes('literal')
        num_clauses += G.number_of_nodes('clause')
        
        curr_l2c = G['l2c'].adjacency_matrix().t() #FIXME: remove the .t() for DGL version 0.5
        curr_aagf = G['aag_forward'].adjacency_matrix().t() #FIXME: remove the .t() for DGL version 0.5
        curr_lit_labels = G.nodes['literal'].data['lit_labels']
        curr_clause_labels = G.nodes['clause'].data['clause_labels']
        
        l2c_indices_0 += (curr_l2c._indices()[0] + shift_lits).tolist()
        l2c_indices_1 += (curr_l2c._indices()[1] + shift_clauses).tolist()
        
        aagf_indices_0 += (curr_aagf._indices()[0] + shift_lits).tolist()
        aagf_indices_1 += (curr_aagf._indices()[1] + shift_lits).tolist()
        
        lit_labels = torch.cat([lit_labels, curr_lit_labels], dim=0)
        clause_labels = torch.cat([clause_labels, curr_clause_labels], dim=0)

#    print('LOOP took {} seconds'.format(time.time()-t2))     

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
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/a.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/a.qaiger.qdimacs')
#b = CombinedGraph1Base()
#b.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/b.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/b.qaiger.qdimacs')
#c = CombinedGraph1Base()
#c.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/c.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/c.qaiger.qdimacs')
#A, B, C = a.G, b.G, c.G
#G = batched_combined_graph([A, B, C])
##A, B = a.G, b.G
##G = batched_combined_graph2([A, B])
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
#print('batching 3 took {} seconds'.format(time.time()-t1))