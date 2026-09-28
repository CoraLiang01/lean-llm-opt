import gurobipy as gp
from gurobipy import GRB
nodes = ['Depot', 'A', 'B', 'C']
customers = ['A', 'B', 'C']
distance = {'Depot': {'Depot': 0, 'A': 28, 'B': 41, 'C': 63}, 'A': {'Depot': 28, 'A': 0, 'B': 27, 'C': 87}, 'B': {'Depot': 41, 'A': 27, 'B': 0, 'C': 81}, 'C': {'Depot': 63, 'A': 87, 'B': 81, 'C': 0}}
for i in nodes:
    for j in nodes:
        if i != j and j not in distance[i]:
            raise ValueError(f'Missing distance from {i} to {j}')
m = gp.Model('Courier_TSP')
x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
u = m.addVars(customers, vtype=GRB.INTEGER, lb=1, ub=3, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
for i in customers:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
    m.addConstr(gp.quicksum((x[j, i] for j in nodes if j != i)) == 1, name=f'in_{i}')
m.addConstr(gp.quicksum((x['Depot', j] for j in nodes if j != 'Depot')) == 1, name='depot_out')
m.addConstr(gp.quicksum((x[i, 'Depot'] for i in nodes if i != 'Depot')) == 1, name='depot_in')
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'subtour_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')