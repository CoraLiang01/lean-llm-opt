import gurobipy as gp
from gurobipy import GRB
tasks = [str(i) for i in range(1, 41)]
cpus = ['1', '2', '3']
b = {'1': 1.1, '2': 2.1, '3': 3, '4': 1, '5': 0.7, '6': 5, '7': 3, '8': 3.5, '9': 4.4, '10': 3.8, '11': 3.5, '12': 2.8, '13': 4.1, '14': 2.9, '15': 5.4, '16': 5.8, '17': 2.6, '18': 4.9, '19': 3.4, '20': 3.6, '21': 5.6, '22': 0.9, '23': 1, '24': 0.6, '25': 5.1, '26': 4.8, '27': 5.3, '28': 5.9, '29': 4.9, '30': 3, '31': 4.8, '32': 1.2, '33': 4, '34': 1.3, '35': 5.7, '36': 3.4, '37': 2.8, '38': 2, '39': 4.8, '40': 3}
f = {'1': 1.33, '2': 2, '3': 2.66}
for i in tasks:
    if i not in b:
        raise ValueError(f'Missing b for task {i}')
for j in cpus:
    if j not in f:
        raise ValueError(f'Missing f for cpu {j}')
p = {}
for i in tasks:
    p[i] = {}
    for j in cpus:
        p[i][j] = b[i] / f[j]
m = gp.Model('TaskAssignment')
x = m.addVars(tasks, cpus, vtype=GRB.BINARY, name='')
Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Cmax')
m.setObjective(Cmax, GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in cpus)) == 1 for i in tasks), name='')
m.addConstrs((gp.quicksum((p[i][j] * x[i, j] for i in tasks)) <= Cmax for j in cpus), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')