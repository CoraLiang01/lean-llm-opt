import gurobipy as gp
from gurobipy import GRB
nodes = ['Depot', 'A', 'B', 'C']
customers = ['A', 'B', 'C']
distance = {'Depot': {'Depot': 0, 'A': 28, 'B': 41, 'C': 63}, 'A': {'Depot': 28, 'A': 0, 'B': 27, 'C': 87}, 'B': {'Depot': 41, 'A': 27, 'B': 0, 'C': 81}, 'C': {'Depot': 63, 'A': 87, 'B': 81, 'C': 0}}
for i in nodes:
    if i not in distance or set(distance[i].keys()) != set(nodes):
        raise ValueError(f'Distance data missing for node {i}')
m = gp.Model('Courier_TSP')
x_vars = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
u_vars = m.addVars(customers, vtype=GRB.INTEGER, lb=1, ub=3, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in nodes if j != i)) == 1 for i in nodes), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in nodes if i != j)) == 1 for j in nodes), name='')
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')