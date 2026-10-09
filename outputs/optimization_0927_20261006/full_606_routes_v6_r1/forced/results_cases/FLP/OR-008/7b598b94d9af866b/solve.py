import gurobipy as gp
from gurobipy import GRB
warehouses = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
stores = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
demand = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
supply = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
cost = {'Supplier1': {'Customer1': 2, 'Customer2': 3, 'Customer3': 1, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier2': {'Customer1': 1, 'Customer2': 2, 'Customer3': 3, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier3': {'Customer1': 3, 'Customer2': 1, 'Customer3': 2, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}, 'Supplier4': {'Customer1': 2, 'Customer2': 3, 'Customer3': 2, 'Customer4': 1, 'Customer5': 3, 'Customer6': 4}, 'Supplier5': {'Customer1': 3, 'Customer2': 2, 'Customer3': 3, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}}
for i in warehouses:
    if i not in cost or i not in supply:
        raise ValueError(f'Missing cost or supply data for warehouse {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for warehouse {i}, store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand data for store {j}')
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= supply[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')