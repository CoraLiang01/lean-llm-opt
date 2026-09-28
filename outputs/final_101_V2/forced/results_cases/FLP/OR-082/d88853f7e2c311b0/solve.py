import gurobipy as gp
from gurobipy import GRB
locations = ['Depot', 'A', 'B', 'C']
distance = {'Depot': {'Depot': 0, 'A': 28, 'B': 41, 'C': 63}, 'A': {'Depot': 28, 'A': 0, 'B': 27, 'C': 87}, 'B': {'Depot': 41, 'A': 27, 'B': 0, 'C': 81}, 'C': {'Depot': 63, 'A': 87, 'B': 81, 'C': 0}}
for i in locations:
    for j in locations:
        if i != j and j not in distance[i]:
            raise ValueError(f'Missing distance from {i} to {j}')
customers = ['A', 'B', 'C']
m = gp.Model('TSP_Van')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=GRB.BINARY, name='')
u = {}
for loc in locations:
    if loc == 'Depot':
        u[loc] = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name='uDepot')
    else:
        u[loc] = m.addVar(lb=1, ub=3, vtype=GRB.CONTINUOUS, name=f'u{loc}')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'depart_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'subtour_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')