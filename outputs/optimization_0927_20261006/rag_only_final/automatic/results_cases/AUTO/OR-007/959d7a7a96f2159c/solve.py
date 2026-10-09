import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5']
stores = ['D1', 'D2', 'D3', 'D4', 'D5']
demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
supply = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
cost = {'S1': {'D1': 269.39105880208, 'D2': 1.45373353909, 'D3': 99.60345345757, 'D4': 26.6407816631, 'D5': 9.53768895688}, 'S2': {'D1': 9.29184687679, 'D2': 10.87477843707, 'D3': 144.52609291615, 'D4': 11.4201330779, 'D5': 153.17568199278}, 'S3': {'D1': 9.67458430167, 'D2': 2.61916509597, 'D3': 100.82422491687, 'D4': 3.21219108879, 'D5': 133.84933961242}, 'S4': {'D1': 270.5749848001, 'D2': 32.50253586, 'D3': 4.68420980965, 'D4': 1.56822696865, 'D5': 9.58927599}, 'S5': {'D1': 226.03319106758, 'D2': 8.66916198083, 'D3': 65.47681316968, 'D4': 9.06876525846, 'D5': 202.65015316426}}
for i in warehouses:
    if i not in cost:
        raise ValueError(f'Missing cost row for warehouse {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost entry for warehouse {i}, store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
for i in warehouses:
    if i not in supply:
        raise ValueError(f'Missing supply for warehouse {i}')
m = gp.Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
for j in stores:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j], name='d_' + j)
for i in warehouses:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in stores)) <= supply[i], name='s_' + i)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in warehouses:
        for j in stores:
            v = x_vars[i, j]
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')