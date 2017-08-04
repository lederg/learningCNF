import subprocess

# def randomCNF():
fuzz = subprocess.Popen("./fuzzsat-0.1/fuzzsat -i 3 -I 500 -p 10 -P 20", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output = fuzz.stdout.readlines()

occs = {}
maxvar = None

def is_number(s):
    try:
        int(s)
    except ValueError:
        return False

    return True

for line in output:
    lits = line.split()[0:-1]
    if is_number(lits[0]):
        lits = map(int,lits)
        for l in lits:
            if abs(l) not in occs:
                occs[abs(l)] = []
            occs[abs(l)] += [lits] # storing reference to lits so we can manipulate them consistently
    else:
        if lits[0] == 'p':
            maxvar = int(lits[2])
            
assert(maxvar != None)

# define sign, as we need it in the next for loop
sign = lambda x: x and (1, -1)[x < 0]

# Split variables when they occur in more than 8 clauses
itervars = set(occs.keys())
added_vars = 0
while True:
    if len(itervars) == 0:
        break
    v = itervars.pop()
    if len(occs[v]) > 8:
        # print('Found var '+str(v)+ ' with '+ str(len(occs[v]))+ ' occurrences.')
        maxvar += 1
        added_vars += 1
        glueclauses = [[v,-maxvar],[-v,maxvar]]
        ## prepend glueclauses to shift all clauses back, don't want to remove the glueclauses with what follows
        occs[v] = glueclauses + occs[v] 
        assert(len(occs[v][10:]) > 0) # > 8 and we added the two glueclauses
        
        assert(maxvar not in occs)
        occs[maxvar] = glueclauses
        
        # move surplus clauses over to new variable
        for clause in occs[v][8:]:
            # change clause inplace, so change is consistent for occurrence lists of other variables
            clause[:] = map(lambda x: maxvar * sign(x) if abs(x) == v else x, clause) 
            occs[maxvar] += [clause]
        assert(len(occs[v]) > len(occs[maxvar]))
        
        occs[v] = occs[v][:7]
        
        # if len(occs[maxvar]) > 8: 
            # print('  new var '+str(maxvar)+ ' will be back.')
        
        itervars.add(maxvar)
        

print ('maxvar: ' + str(maxvar))
print ('added vars: ' + str(added_vars))
print ('Max: ' + str( max( [len(occs[v]) for v in occs.keys()] )))
print ('Over 8 occs: ' + str(len( filter(lambda x: x, [len(occs[v]) > 8 for v in occs.keys()] ))))

clauses = set()
clause_list = []
for v in occs.keys():
    for c in occs[v]:
        c_string = ' '.join(map(str,c))
        if c_string not in clauses:
            clauses.add(c_string)
            clause_list.append(c_string)

textfile = open("out.dimacs", "w")
textfile.write('p cnf ' + str(maxvar) + ' ' + str(len(clause_list)) + '\n')
for c in clause_list:
    textfile.write(c + ' 0\n')
textfile.close()


textfile = open("tmp.dimacs", "w")
textfile.writelines(output)
textfile.close()

sat = subprocess.Popen("picosat tmp.dimacs", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sat.communicate()[0]
print(sat.returncode)

sat2 = subprocess.Popen("picosat out.dimacs", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sat2.communicate()[0]
print(sat2.returncode)
