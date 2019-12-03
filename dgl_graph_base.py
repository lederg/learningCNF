from cnf_parser import qdimacs_to_cnf
from aag_parser import read_qaiger, read_wordlevel
#from utils import *
#from torch.utils.data import Dataset
import torch
import dgl

class DGL_Graph_Base(object):    
    """
    Wrapper object for paired files ('a.wordlevel', 'a.qaiger', 'a.qdimacs') 
    Holds the combined Wordlevel-AAG-CNF graph    
    Uses the DGL package to store this heterogeneous graph
        Node Types: 
            if var_graph == FALSE (default):
                literals
            if var_graph == TRUE:
                variables
            clauses
            wordlevel nodes: TBD
        Edge types: 
            AAG literal -> AAG literal forward edges 
            AAG literal -> AAG literal backward edges 
            QCNF literal -> QCNF clause forward edges 
            QCNF clause -> QCNF literal backward edges
            Wordlevel edges: TBD
        9-dimensional features for literal nodes
        1-dimensional features for clause nodes
    """
    def __init__(self, qcnf = None, qcnf_base = None, aag = None, wl = None, G = None, var_graph=False, **kwargs):
        self.qcnf = qcnf # WARNING: if use update_DGL_graph(), check that fix_qcnf_numbering() is used too
        self.qcnf_base = qcnf_base
        self.aag = aag
        self.wl = wl
        self.aag_forward_edges = []
        self.aag_backward_edges = []
        self.lit_labels = None
        
        self.aag_f_p, self.aag_f_n, self.aag_b_p, self.aag_b_n = [], [], [], []
        self.v2c_p, self.v2c_n, self.c2v_p, self.c2v_n = [], [], [], []
        self.var_labels = None

        self.G = G
        self.extra_clauses = {}
        self.removed_old_clauses = []
        self.LIT_FEATURE_DIM = 9
        self.CLAUSE_FEATURE_DIM = 1
        self.IDX_VAR_UNIVERSAL = 0
        self.IDX_VAR_EXISTENTIAL = 1
        self.IDX_VAR_INPUT_OUTPUT = 8 # Last Column
        self.var_graph = var_graph
        
    def reset(self):
        self.extra_clauses = {}
        self.removed_old_clauses = []
        
    def load_paired_files(self, wordlevel_fname = None, aag_fname = None, qcnf_fname = None, var_graph=False):
        # self.aag_og = read_qaiger(aag_fname)        # for testing
        # self.qcnf_og = qdimacs_to_cnf(qcnf_fname)   # for testing
        if not qcnf_fname: raise Exception("need a qcnf graph; see dgl_graph_base.py")
        self.var_graph = var_graph
        if wordlevel_fname: self.wl = read_wordlevel(wordlevel_fname)
        self.aag = fix_aag_numbering(read_qaiger(aag_fname)) if aag_fname else None
        self.qcnf = fix_qcnf_numbering(qdimacs_to_cnf(qcnf_fname))
        self.reset()
        self.create_DGL_graph()
        
    def create_DGL_graph(self, qcnf_base = None, old_base = None,
            l2w_edges = [], w2l_edges = [],
            extra_clauses = {}, removed_old_clauses = [], 
        ):
        """
        Create the combined AAG-QCNF graph.
        
        Updates from CADET:
            extra_clauses : learned/derived clauses not included in original qcnf formula
                add clause node
                add clause node embeddings `1`
                add l2c edges
                add c2l edges
            removed_old_clauses : clauses included in original qcnf formula to be removed
                remove l2c edges
                remove c2l edges
        """
        if qcnf_base:
            self.qcnf_base = qcnf_base
            
        ######## edges before CADET update
        if old_base: # creating this graph from an older graph
            old_G = old_base.G
            if self.var_graph: # variable graph
                aag_f_p, aag_f_n, aag_b_p, aag_b_n = old_base.aag_f_p, old_base.aag_f_n, old_base.aag_b_p, old_base.aag_b_n
                v2c_p, v2c_n, c2v_p, c2v_n = old_base.v2c_p, old_base.v2c_n, old_base.c2v_p, old_base.c2v_n
                var_labels = old_G.nodes['variable'].data['var_labels']
            else: # literal graph
                aag_forward_edges, aag_backward_edges = old_base.aag_forward_edges, old_base.aag_backward_edges
                qcnf_forward_edges, qcnf_backward_edges = old_base.qcnf_forward_edges, old_base.qcnf_backward_edges
                lit_labels = old_G.nodes['literal'].data['lit_labels']
        else: # create this graph from scratch
            clause_labels =  self.initial_clause_features()
            if self.var_graph: # variable graph
                aag_f_p, aag_f_n, aag_b_p, aag_b_n = self.initial_aag_edges()
                v2c_p, v2c_n, c2v_p, c2v_n = self.initial_qcnf_edges()
                var_labels = self.initial_lit_features()
            else: # literal graph
                aag_forward_edges, aag_backward_edges = self.initial_aag_edges()
                qcnf_forward_edges, qcnf_backward_edges = self.initial_qcnf_edges()
                lit_labels = self.initial_lit_features()
            
            
        ######## use CADET update
        ## add qcnf edges for each clause in extra_clauses
        for cl_num in extra_clauses:
            for lit in extra_clauses[cl_num]:            
                if self.var_graph: # graph on variables
                    var = self.node_num(lit)
                    if self.is_positive(lit):
                        v2c_p.append( (var, cl_num) )
                        c2v_p.append( (cl_num, var) )
                    else:
                        v2c_n.append( (var, cl_num) )
                        c2v_n.append( (cl_num, var) )
                else: # graph on literals
                    qcnf_forward_edges.append( (lit, cl_num) )
                    qcnf_backward_edges.append( (cl_num, lit) )
                
        ## change clause embeddings : 1 for extra clauses        
        if extra_clauses:
            clause_labels = torch.zeros([self.num_clauses(extra_clauses), self.CLAUSE_FEATURE_DIM])
            clause_labels[[cl_num for cl_num in extra_clauses.keys()]] = 1

        ## remove qcnf edges for each clause in remove_old_clauses
        for cl_num in removed_old_clauses:
            if self.var_graph: # graph on variables
                v2c_p_cl_num = [e for e in v2c_p if e[1]==cl_num] #v2c
                c2v_p_cl_num = [e for e in c2v_p if e[0]==cl_num] #c2v
                v2c_n_cl_num = [e for e in v2c_n if e[1]==cl_num] #v2c
                c2v_n_cl_num = [e for e in c2v_n if e[0]==cl_num] #c2v
                for e in v2c_p_cl_num:
                    v2c_p.remove(v2c_p_cl_num)
                    c2v_p.remove(c2v_p_cl_num)
                for e in v2c_n_cl_num:
                    v2c_n.remove(v2c_n_cl_num)
                    c2v_n.remove(c2v_n_cl_num)
            else: # graph on literals
                qcnf_forward_edges_cl_num = [e for e in qcnf_forward_edges if e[1]==cl_num] #l2c
                qcnf_backward_edges_cl_num = [e for e in qcnf_backward_edges if e[0]==cl_num] #c2l
                for e in qcnf_forward_edges_cl_num:
                    qcnf_forward_edges.remove(qcnf_forward_edges_cl_num)
                    qcnf_backward_edges.remove(qcnf_backward_edges_cl_num)
                
        ######## Save graph info in the base object
        self.extra_clauses = extra_clauses
        self.removed_old_clauses = removed_old_clauses
        self.clause_labels = clause_labels
        if self.var_graph: # variable graph
            self.var_labels = var_labels
            self.aag_f_p, self.aag_f_n, self.aag_b_p, self.aag_b_n = aag_f_p, aag_f_n, aag_b_p, aag_b_n
            self.v2c_p, self.v2c_n, self.c2v_p, self.c2v_n = v2c_p, v2c_n, c2v_p, c2v_n
        else: # literal graph
            self.lit_labels = lit_labels
            self.qcnf_forward_edges = qcnf_forward_edges
            self.qcnf_backward_edges = qcnf_backward_edges
        
        ######## Form the DGL graph (no word-level)
        if not self.wl:
            if self.var_graph: # variable graph, no word-level
                G = dgl.heterograph(
                        {('variable', 'aag_f_+', 'variable') : aag_f_p,
                         ('variable', 'aag_f_-', 'variable') : aag_f_n,
                         ('variable', 'aag_b_+', 'variable') : aag_b_p,
                         ('variable', 'aag_b_-', 'variable') : aag_b_n,
                         ('variable', 'v2c_+', 'clause') : v2c_p,
                         ('variable', 'v2c_-', 'clause') : v2c_n,
                         ('clause', 'c2v_+', 'variable') : c2v_p,
                         ('clause', 'c2v_-', 'variable') : c2v_n},
                        {'variable': int(self.num_lits/2),
                         'clause': self.num_clauses(extra_clauses)}) 
                G.nodes['variable'].data['var_labels'] = var_labels
                G.nodes['clause'].data['clause_labels'] = clause_labels
                self.G = G
            else: # literal graph, no word-level
                G = dgl.heterograph(
                    {('literal', 'aag_forward', 'literal') : aag_forward_edges,
                     ('literal', 'aag_backward', 'literal') : aag_backward_edges,
                     ('literal', 'l2c', 'clause') : qcnf_forward_edges,
                     ('clause', 'c2l', 'literal') : qcnf_backward_edges},
                    {'literal': self.num_lits,
                     'clause': self.num_clauses(extra_clauses)}) 
                G.nodes['literal'].data['lit_labels'] = lit_labels
                G.nodes['clause'].data['clause_labels'] = clause_labels
                self.G = G
                
        #### Word Level stuff #################################################
        if self.wl:
            num_vars, num_const, num_intm = self.wl['nodes']['variables'], self.wl['nodes']['constants'], self.wl['nodes']['intermediates']    
            num_wl_nodes = num_vars + num_const + num_intm
            
            if not l2w_edges or not w2l_edges:
                if self.aag:
                    l2w_edges, w2l_edges = self.bitblast_edges(aag=True)
                else:
                    raise Exception("Not yet implemented; need to figure out bitblasting without aag circuit...")
            
            wl_op_edges_f = []
            for e in self.wl['f_edges'].values(): 
                wl_op_edges_f += e
            wl_op_edges_b = []
            for e in self.wl['b_edges'].values(): 
                wl_op_edges_b += e
                
            ###### Form the DGL graph (word-level)
            G = dgl.heterograph(
                    {('literal', 'aag_forward', 'literal') : aag_forward_edges,
                     ('literal', 'aag_backward', 'literal') : aag_backward_edges,
                     ('literal', 'l2c', 'clause') : qcnf_forward_edges,
                     ('clause', 'c2l', 'literal') : qcnf_backward_edges,
                     
                     ('literal', 'l2w', 'wl_node') : l2w_edges,
                     ('wl_node', 'w2l', 'literal') : w2l_edges,
                     
                     ('wl_node', 'wl_op_f', 'wl_node') : wl_op_edges_f,
                     ('wl_node', 'wl_op_b', 'wl_node') : wl_op_edges_b},
                     
#                     ('wl_node', '+_f', 'wl_node') : self.wl['f_edges']['+'],
#                     ('wl_node', 'and_f', 'wl_node') : self.wl['f_edges']['and'],
#                     ('wl_node', 'or_f', 'wl_node') : self.wl['f_edges']['or'],
#                     ('wl_node', 'xor_f', 'wl_node') : self.wl['f_edges']['xor'],
#                     ('wl_node', 'invert_f', 'wl_node') : self.wl['f_edges']['invert'],
#                     ('wl_node', 'abs_f', 'wl_node') : self.wl['f_edges']['abs'],
#                     ('wl_node', 'neg_f', 'wl_node') : self.wl['f_edges']['neg'],
#                     ('wl_node', '=_f', 'wl_node') : self.wl['f_edges']['='],
#                     ('wl_node', '!=_f', 'wl_node') : self.wl['f_edges']['!='],
#                     ('wl_node', '-L_f', 'wl_node') : self.wl['f_edges']['-L'],
#                     ('wl_node', '-R_f', 'wl_node') : self.wl['f_edges']['-R'],
#                     ('wl_node', '<L_f', 'wl_node') : self.wl['f_edges']['<L'],
#                     ('wl_node', '<R_f', 'wl_node') : self.wl['f_edges']['<R'],
#                     ('wl_node', '<=L_f', 'wl_node') : self.wl['f_edges']['<=L'],
#                     ('wl_node', '<=R_f', 'wl_node') : self.wl['f_edges']['<=R'],
#                     ('wl_node', '>L_f', 'wl_node') : self.wl['f_edges']['>L'],
#                     ('wl_node', '>R_f', 'wl_node') : self.wl['f_edges']['>R'],
#                     ('wl_node', '>=L_f', 'wl_node') : self.wl['f_edges']['>=L'],
#                     ('wl_node', '>=R_f', 'wl_node') : self.wl['f_edges']['>=R'],
#                     
#                     ('wl_node', '+_b', 'wl_node') : self.wl['b_edges']['+'],
#                     ('wl_node', 'and_b', 'wl_node') : self.wl['b_edges']['and'],
#                     ('wl_node', 'or_b', 'wl_node') : self.wl['b_edges']['or'],
#                     ('wl_node', 'xor_b', 'wl_node') : self.wl['b_edges']['xor'],
#                     ('wl_node', 'invert_b', 'wl_node') : self.wl['b_edges']['invert'],
#                     ('wl_node', 'abs_b', 'wl_node') : self.wl['b_edges']['abs'],
#                     ('wl_node', 'neg_b', 'wl_node') : self.wl['b_edges']['neg'],
#                     ('wl_node', '=_b', 'wl_node') : self.wl['b_edges']['='],
#                     ('wl_node', '!=_b', 'wl_node') : self.wl['b_edges']['!='],
#                     ('wl_node', '-L_b', 'wl_node') : self.wl['b_edges']['-L'],
#                     ('wl_node', '-R_b', 'wl_node') : self.wl['b_edges']['-R'],
#                     ('wl_node', '<L_b', 'wl_node') : self.wl['b_edges']['<L'],
#                     ('wl_node', '<R_b', 'wl_node') : self.wl['b_edges']['<R'],
#                     ('wl_node', '<=L_b', 'wl_node') : self.wl['b_edges']['<=L'],
#                     ('wl_node', '<=R_b', 'wl_node') : self.wl['b_edges']['<=R'],
#                     ('wl_node', '>L_b', 'wl_node') : self.wl['b_edges']['>L'],
#                     ('wl_node', '>R_b', 'wl_node') : self.wl['b_edges']['>R'],
#                     ('wl_node', '>=L_b', 'wl_node') : self.wl['b_edges']['>=L'],
#                     ('wl_node', '>=R_b', 'wl_node') : self.wl['b_edges']['>=R']},
                    
                {'literal': self.num_lits,
                 'clause': self.num_clauses(extra_clauses),
                 'wl_node': num_wl_nodes})
            
            G.nodes['literal'].data['lit_labels'] = lit_labels
            G.nodes['clause'].data['clause_labels'] = clause_labels
            self.G = G
            
            #G.nodes['wl_node'].data['wl_node_labels'] = clause_labels
        #######################################################################
            
    def node_num(self, x):
        """
        type(x) == int
        x is a literal number        0,1, 2,3 4,5 ... 2n-2,2n-1
        if self.var_graph == True:
            use variable numberings  0,   1,  2,  ... n 
        """
        return int(x/2) if self.var_graph else x
    
    def is_positive(self, lit):
        """
        lit is a literal 0,1,...,2n-2,2n-1
        return True iff lit is even (positive)
        """
        return lit % 2 == 0

    def bitblast_edges(self, aag): 
        """
        connect the word-level variables to the aag-literals
        """
        if aag:
            vn = self.wl['nodes']['variable_names']
            inp = self.aag['input_symbols']
            l2w, w2l =[], []
            for wl_var in vn:
                for i, bit_name in enumerate(inp):
                    if bit_name.startswith(vn[wl_var]):
                        l2w.append( (i, wl_var) )
                        w2l.append( (wl_var, i) )
            return (l2w, w2l)
    
    def initial_aag_edges(self):
        """
        create aag edges, which remain fixed
        """
        aag_forward_edges, aag_backward_edges = [], []
        aag_f_p, aag_f_n, aag_b_p, aag_b_n = [], [], [], []
        if self.aag:
            for ag in self.aag['and_gates']:
                x, y, z = ag[0], ag[1], ag[2]
                if self.var_graph: # graph on variables
                    vx, vy, vz = self.node_num(ag[0]), self.node_num(ag[1]), self.node_num(ag[2])
                    if self.is_positive(x):
                        aag_f_p.append( (vx,vy) )
                        aag_f_p.append( (vx,vz) )
                    else:
                        aag_f_n.append( (vx,vy) )
                        aag_f_n.append( (vx,vz) )
                    if self.is_positive(y):
                        aag_b_p.append( (vy,vx) )
                    else:
                        aag_b_n.append( (vy,vx) )
                    if self.is_positive(z):
                        aag_b_p.append( (vz,vx) )
                    else:
                        aag_b_n.append( (vz,vx) )
                else: # graph on literals
                    aag_forward_edges.append( (x,y) )
                    aag_forward_edges.append( (x,z) )
                    aag_backward_edges.append( (y,x) )
                    aag_backward_edges.append( (z,x) )
        if self.var_graph: # graph on variables
            
            return (aag_f_p, aag_f_n, aag_b_p, aag_b_n)
        else: # graph on literals
            self.aag_forward_edges = aag_forward_edges
            self.aag_backward_edges = aag_backward_edges
            return (aag_forward_edges, aag_backward_edges)
    
    def initial_qcnf_edges(self):
        """
        create qcnf edges, from original clauses only
        need to be updated from CADET obs
        """
        l2c, c2l = [], []
        v2c_p, v2c_n, c2v_p, c2v_n = [], [], [], []
        for cl_num, clause in enumerate(self.qcnf['clauses']):
            for lit in clause:
                if self.var_graph: # graph on variables
                    var = self.node_num(lit)
                    if self.is_positive(lit):
                        v2c_p.append( (var, cl_num) )
                        c2v_p.append( (cl_num, var) )
                    else:
                        v2c_n.append( (var, cl_num) )
                        c2v_n.append( (cl_num, var) )
                else: # graph on literals
                    l2c.append( (lit, cl_num) )
                    c2l.append( (cl_num, lit) )
        if self.var_graph: # graph on variables
            return (v2c_p, v2c_n, c2v_p, c2v_n)
        else: # graph on literals
            self.qcnf_forward_edges = l2c
            self.qcnf_backward_edges = c2l
            return (l2c, c2l)
    
    def initial_lit_features(self):
        """
        9 dimensional features for 'lit' nodes. Shape (num_literals, num_features=9).
        1st column: 1 iff lit is universal (from qcnf graph)
        2nd column: 1 iff lit is existential (from qcnf graph)
        9th column: 1 iff lit is an input or output (from aag graph)
        """
        universal_lits, existential_lits = [], []
        for v in self.qcnf['cvars'].keys():
            if self.qcnf['cvars'][v]['universal']:
                universal_lits.append(2*v)
                universal_lits.append(2*v+1)
            else:
                existential_lits.append(2*v)
                existential_lits.append(2*v+1)
        embs = torch.zeros([2 * self.qcnf['maxvar'], self.LIT_FEATURE_DIM]) 
        embs[:, self.IDX_VAR_UNIVERSAL][universal_lits] = 1
        embs[:, self.IDX_VAR_EXISTENTIAL][existential_lits] = 1
        if self.aag:
            flipped_inputs = [flip(i) for i in self.aag['inputs']]
            flipped_outputs = [flip(o) for o in self.aag['outputs']]
            embs[:, self.IDX_VAR_INPUT_OUTPUT][self.aag['inputs']] = 1
            embs[:, self.IDX_VAR_INPUT_OUTPUT][flipped_inputs] = 1
            embs[:, self.IDX_VAR_INPUT_OUTPUT][self.aag['outputs']] = 1
            embs[:, self.IDX_VAR_INPUT_OUTPUT][flipped_outputs] = 1
        if self.var_graph:
            return embs[[i for i in range(embs.shape[0]) if i%2==0],:]
        return embs
    
    def initial_clause_features(self):
        """
        1 dimensional features for 'clause' nodes. Shape (num_clauses, num_features=1).
        1st column: 0 if clause is original, 1 if clause is learned/derived (so, initially all 0).
        """
        return torch.zeros([self.num_og_clauses, self.CLAUSE_FEATURE_DIM])
    
    def add_clause(self,clause, clause_id):
        assert(clause_id not in self.extra_clauses.keys())
        self.extra_clauses[clause_id]=clause

    def remove_clause(self, clause_id):
        if not (clause_id in self.extra_clauses.keys()):            
            self.removed_old_clauses.append(clause_id)
        else:
            del self.extra_clauses[clause_id]
            
    def update_ground_embs(self, ground_embs):
        """
        n = num_vars, 2*n = num_lits
        ground_embs has shape (n x 8)
        want to update lit_labels, which has shape (2*n x 9)
        so, for each var, use the 8 dimensional feature as the first 8 entries 
            of the rows of both literal labels...
        """
        n, num_feats = ground_embs.shape[0], ground_embs.shape[1]
        u = torch.cat((ground_embs, ground_embs),dim=1).view(2*n, num_feats)
        if self.var_graph: # variable graph
            self.G.nodes['variable'].data['var_labels'][:,:num_feats] = u
        else: # literal graph
            self.G.nodes['literal'].data['lit_labels'][:,:num_feats] = u
        
##############################################################################
##### Properties    
##############################################################################        
    @property
    def num_vars(self):
        return self.qcnf['maxvar']
    
    @property
    def num_lits(self):
        return 2 * self.num_vars
    
#    @property
#    def num_clauses(self): ##FIXME ??
#        k = self.extra_clauses.keys()
#        if not k:
#            return self.num_og_clauses     
#        return max(k)+1
        
    def num_clauses(self, extra_clauses): ##FIXME ??
        k = extra_clauses.keys()
        if not k:
            return self.num_og_clauses     
        return max(k)+1
    
    @property
    def num_og_clauses(self): 
        return self.qcnf['num_clauses'] 
    
##############################################################################
##### Functions involving QCNF, AAG, Literal/Variable numbering    
##############################################################################
        
def fix_aag_numbering(aag):
    """
    .qaiger file read in with variables 1,2,...,n and literals 2,3,4,5,...,2n,2n+1.
    Change  literals 2,  3, 4,  5, ..., 2n,  2n+1
    to      literals 0,  1, 2,  3, ..., 2n-2,2n-1 
    in order to make the literals 0-based.
    """
    for i, inp in enumerate(aag['inputs']):
        aag['inputs'][i] = int(inp - 2)            
    for i, out in enumerate(aag['outputs']):
        aag['outputs'][i] = int(out - 2)
    for ag in aag['and_gates']:
        for i, e in enumerate(ag):
            ag[i] = int(e - 2)
    return aag

def convert_qdimacs_lit(lit):
    L = 2 * abs(lit)
    L = L + 1 if (lit < 0) else L
    return int(L - 2)

def fix_qcnf_numbering(qcnf):
    """
    .qdimacs file read in with variables 1,2,...,n and literals 1,-1,2,-2,...,n,-n.
    Change  literals 1, -1, 2, -2, ..., n,   -n
    to      literals 2,  3, 4,  5, ..., 2n,  2n+1
    to      literals 0,  1, 2,  3, ..., 2n-2,2n-1 
    in order to make the literals 0-based.
    """
    # change literal numbers: in 'clauses'
    for cl in qcnf ['clauses']:
        for i, lit in enumerate(cl):
            cl[i] = convert_qdimacs_lit(lit)
            
    # change variable numbers: keys of 'cvars'
    cvars = {}
    for key in qcnf['cvars']:
        cvars[int(key - 1)] = qcnf['cvars'][key]
    qcnf['cvars'] = cvars
    
    return qcnf

def flip(lit):
    """
    literals 0, 1, 2, 3, ..., 2n-2, 2n-1 
    if lit is even (positive literal),  not_lit = lit + 1
    if lit is odd (negative literal),   not_lit = lit - 1
    """
    return lit + 1 if lit % 2 == 0 else lit - 1

def vars_to_lits(V, sign=None):
    """
    Given list of variables V = [v0, v1, ..., vn] (all positive integers 0,1,2,...),
    return list of literals [2*v0, 2*v0+1, 2*v1, 2*v1+1, ..., 2*vn, 2*vn+1]
    """
    L = []
    for v in V:
        if not sign:
            L.append(int(2*v))
            L.append(int(2*v+1))
        if sign == 1:
            L.append(int(2*v))
        elif sign == -1:
            L.append(int(2*v+1))
    return L

def Vars01_to_Lits01(V):
    """
    Given list in {0,1} indexed by variable (size n=num_vars),
    return list in {0,1} indexed by literal (size 2n=num_lits),
    where literal entry is 1 if corresponding variable entry is 1, else 0.
    """
    L = []
    for v in V:
        if v == 1:
            L.append(1)
            L.append(1)
        else:
            L.append(0)
            L.append(0)
    return torch.tensor(L)
    
    
##############################################################################
##### TESTING     
#b = DGL_Graph_Base()
#b.load_paired_files(
#        wordlevel_fname = './data/words_3_levels_1/words_2.wordlevel', 
#        aag_fname = './data/words_3_levels_1/words_2.qaiger', 
#        qcnf_fname = './data/words_3_levels_1/words_2.qaiger.qdimacs')
    
c = DGL_Graph_Base()
c.load_paired_files(
        aag_fname = './data/words_test_ryan_0/words_0_SAT.qaiger', 
        qcnf_fname = './data/words_test_ryan_0/words_0_SAT.qaiger.qdimacs')
    
d = DGL_Graph_Base()
d.load_paired_files(
        aag_fname = './data/words_test_ryan_0/words_0_SAT.qaiger', 
        qcnf_fname = './data/words_test_ryan_0/words_0_SAT.qaiger.qdimacs',
        var_graph =  True)