import json
from pprint import pprint

eq_classes = []

with open('expressions-synthetic/largeSimpleBoolean5.json') as json_file:
    
    json = json.load(json_file)
    
    eq_class_keys = json.keys()
    
    for k in eq_class_keys:
        print(k)
        eq_class = []
        eq_class.append(json[k]['Original']['Tree']['Children']['child'])
        num_formulas = 1
        for j in range(len(json[k]['Noise'])):
            eq_class.append(json[k]['Noise'][j]['Tree']['Children']['child'])
            num_formulas += 1
        print('  ' + str(num_formulas) + ' variants')
        eq_classes.append(eq_class)
    del eq_class_keys
    del json


def simplify_clause(c):
    s = set(c)
    for x in s:
        if -x in s:
            return None
    return list(s)

def add_clauses(CNF, clauses):
    for c in clauses:
        c = simplify_clause(c)
        if c != None:
            CNF['clauses'] += [c]
            for l in c:
                v = abs(l)
                if v not in CNF['clauses_per_variable']:
                    CNF['clauses_per_variable'][v] = []
                CNF['clauses_per_variable'][v] += [c]
            

def translate_expression(CNF, node):
    name = node['Name']
    if len(name) == 1 and name.islower(): # Variable
        if name not in CNF['origvars']:
            CNF['maxvar'] += 1
            CNF['origvars'][name] = CNF['maxvar']
        CNF['topvar'] = CNF['origvars'][name]
        
    elif name == 'Not':
        subexpr = - translate_expression(CNF, node['Children']['child'])
        CNF['topvar'] = subexpr
        return subexpr
        
        # Alternative encoding for Not, introducing a new variable
#         subexpr = translate_expression(CNF, node['Children']['child'])
#         CNF['maxvar'] += 1
#         CNF['topvar'] = CNF['maxvar']
#         CNF['auxvars'] += CNF['maxvar']
#         add_clauses(CNF, [[subexpr, CNF['maxvar']],[- subexpr, - CNF['maxvar']])
#         return CNF['maxvar']
        
    elif name == 'Or':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [- subexpr_left,  CNF['topvar']], \
                            [- subexpr_right, CNF['topvar']], \
                            [subexpr_left, subexpr_right, - CNF['topvar']]
                        ])
        
    elif name == 'Implies':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [  subexpr_left,  CNF['topvar']], \
                            [- subexpr_right, CNF['topvar']], \
                            [- subexpr_left, subexpr_right, - CNF['topvar']]
                        ])
        
    elif name == 'And':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [subexpr_left,  - CNF['topvar']], \
                            [subexpr_right, - CNF['topvar']], \
                            [ - subexpr_left, - subexpr_right, CNF['topvar']]
                        ])
        
    elif name == 'Xor':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [  subexpr_left,   subexpr_right,  - CNF['topvar']], \
                            [- subexpr_left, - subexpr_right,  - CNF['topvar']], \
                            [  subexpr_left, - subexpr_right,    CNF['topvar']], \
                            [- subexpr_left,   subexpr_right,    CNF['topvar']], \
                        ])
        
    else:
        print('Error: Unknown operator ' + name)
        quit() 
        
    return CNF['topvar']

# Translate all formulas to CNF
eq_classes_CNF = {}
for eq_class in eq_classes:
    eq_class_CNF = []
    for formula_node in eq_class:
        print('Processing: ')
        pprint(formula_node)
        CNF = {'topvar' : None, 'maxvar' : 0, 'origvars': {}, 'auxvars': [], 'clauses': [], 'clauses_per_variable' : {}}
        eq_class_CNF += [CNF]        
        translate_expression(CNF, formula_node)
        print(CNF)
        
        for v in CNF['clauses_per_variable'].keys():
            if len(CNF['clauses_per_variable'][v]) > 8:
                print('Error: too many clauses for variable ' + str(v))
                quit()
        
    eq_classes_CNF[formula_node['Symbol']] = eq_class_CNF 
    

