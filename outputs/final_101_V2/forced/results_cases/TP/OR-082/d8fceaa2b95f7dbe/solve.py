import gurobipy as gp
from gurobipy import GRB
nodes = ['Depot', 'A', 'B', 'C']
distance = {'Depot': {'Depot': 0, 'A': 28, 'B': 41, 'C': 63}, 'A': {'Depot': 28, 'A': 0, 'B': 27, 'C': 87}, 'B': {'Depot': 41, 'A': 27, 'B': 0, 'C': 81}, 'C': {'Depot': 63, 'A': 87, 'B': 81, 'C': 0}}
for i in nodes:
    for j in nodes:
        if i == j:
            continue
        if i not in distance or j not in distance[i]:
            raise ValueError(f'Missing distance from {i} to {j}')
m = gp.Model('Courier_TSP')
x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
u = {}
for n in nodes:
    if n == 'Depot':
        u[n] = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name=f'u_{n}')
    else:
        u[n] = m.addVar(lb=1, ub=3, vtype=GRB.CONTINUOUS, name=f'u_{n}')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
for i in nodes:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'depart_{i}')
for j in nodes:
    m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'arrive_{j}')
customers = ['A', 'B', 'C']
for i in customers:
    for j in customers:
        if i == j:
            continue
        m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')