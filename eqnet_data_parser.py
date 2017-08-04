import json
from pprint import pprint

eq_classes = []

with open('expressions-synthetic/boolean5.json') as json_file:
    
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


# TODO: translate to CNF

for x in eq_classes[0]:
    pprint(x['Name'])