"""
Batch a list of AAG-CNF graphs
"""
import torch
import dgl
import networkx

def batched_combined_graph(L):
    """L is a list of at least 1 DGL Heterograph"""
    
    num_lits = [0]
    num_clauses = [0]
    for j, G in enumerate(L):
        num_lits.append(G.number_of_nodes('literal') + num_lits[j])
        num_clauses.append(G.number_of_nodes('clause') + num_clauses[j])
    
    l2c_adj = L[0]['l2c'].adjacency_matrix().t() #FIXME: remove the .t() for DGL version 0.5
    aagf_adj = L[0]['aag_forward'].adjacency_matrix().t() #FIXME: remove the .t() for DGL version 0.5
    lit_labels = L[0].nodes['literal'].data['lit_labels']
    
    if len(L) > 1:
        for j, G in enumerate(L[1:]):
            curr_l2c = G['l2c'].adjacency_matrix().t() #FIXME: remove the .t() for DGL version 0.5
            curr_aagf = G['aag_forward'].adjacency_matrix().t() #FIXME: remove the .t() for DGL version 0.5
            curr_lit_labels = G.nodes['literal'].data['lit_labels']
            
            l2c_adj = combine_sparse_adj(l2c_adj, curr_l2c, num_lits[j+1], num_clauses[j+1], num_lits[j+2], num_clauses[j+2]) #FIXME: take transpose for DGL version 0.5
            aagf_adj = combine_sparse_adj(aagf_adj, curr_aagf, num_lits[j+1], num_lits[j+1], num_lits[j+2], num_lits[j+2])
            lit_labels = torch.cat([lit_labels, curr_lit_labels], dim=0)
            
    G = dgl.heterograph(
                {('literal', 'aag_forward', 'literal') : format_edges(aagf_adj),
                 ('literal', 'aag_backward', 'literal') : format_edges(aagf_adj.t()),
                 ('literal', 'l2c', 'clause') : format_edges(l2c_adj),
                 ('clause', 'c2l', 'literal') : format_edges(l2c_adj.t())},
                {'literal': num_lits[len(num_lits)-1],
                 'clause': num_clauses[len(num_clauses)-1]}
    )
    G.nodes['literal'].data['lit_labels'] = lit_labels
    return G
        
def combine_sparse_adj(A, B, shift0, shift1, size0, size1):
    """
    A,B are torch.sparse_coo matrices
    add SHIFT0 to the dim-0 indices of B
    add SHIFT1 to the dim-1 indices of B
    Make A,B have shape [SIZE0, SIZE1]
    Return A + B to create a new sparse matrix of shape [SIZE0, SIZE1]
    """
    B0_indices = B._indices()[0] + shift0
    B1_indices = B._indices()[1] + shift1
    i = torch.cat([B0_indices, B1_indices], dim=0).view(2,-1)
    v = B._values()
    s = torch.Size([size0, size1])
    B_shifted = torch.sparse.FloatTensor(i, v, s)
    i = A._indices()
    v = A._values()
    A_stretched = torch.sparse.FloatTensor(i, v, s)
    return A_stretched + B_shifted

def format_edges(spm):
    return (spm._indices()[0].tolist(), spm._indices()[1].tolist())

###############################################################################
    ### TEST
###############################################################################
a = CombinedGraph1Base()
a.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/a.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/a.qaiger.qdimacs')
b = CombinedGraph1Base()
b.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/b.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/b.qaiger.qdimacs')
c = CombinedGraph1Base()
c.load_paired_files(aag_fname = './data/words_test_ryan_mini_m/c.qaiger', qcnf_fname = './data/words_test_ryan_mini_m/c.qaiger.qdimacs')
#A, B, C = a.G, b.G, c.G
#G = batched_combined_graph([A, B, C])
A, B = a.G, b.G
G = batched_combined_graph([A, B])