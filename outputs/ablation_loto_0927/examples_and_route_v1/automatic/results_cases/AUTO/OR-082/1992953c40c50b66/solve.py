import gurobipy as gp
from gurobipy import GRB
locations = ['Depot', 'A', 'B', 'C']
distance = {'Depot': {'Depot': 0, 'A': 28, 'B': 41, 'C': 63}, 'A': {'Depot': 28, 'A': 0, 'B': 27, 'C': 87}, 'B': {'Depot': 41, 'A': 27, 'B': 0, 'C': 81}, 'C': {'Depot': 63, 'A': 87, 'B': 81, 'C': 0}}
for i in locations:
    for j in locations:
        if i != j and j not in distance[i]:
            raise ValueError(f'Missing distance from {i} to {j}')
arcs = [(i, j) for i in locations for j in locations if i != j]
u_nodes = [loc for loc in locations if loc != 'Depot']
m = gp.Model('TSP_Courier')
x = m.addVars(arcs, vtype=GRB.BINARY, lb=0, name='')
u = m.addVars(u_nodes, lb=1, ub=3, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in arcs)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in locations if j != i)) == 1 for i in locations), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in locations if i != j)) == 1 for j in locations), name='')
for i in u_nodes:
    for j in u_nodes:
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