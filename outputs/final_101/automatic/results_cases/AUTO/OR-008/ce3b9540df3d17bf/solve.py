import gurobipy as gp
from gurobipy import GRB
suppliers = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
customers = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
demand = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
supply_capacity = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
cost = {'Supplier1': {'Customer1': 2, 'Customer2': 3, 'Customer3': 1, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier2': {'Customer1': 1, 'Customer2': 2, 'Customer3': 3, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier3': {'Customer1': 3, 'Customer2': 1, 'Customer3': 2, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}, 'Supplier4': {'Customer1': 2, 'Customer2': 3, 'Customer3': 2, 'Customer4': 1, 'Customer5': 3, 'Customer6': 4}, 'Supplier5': {'Customer1': 3, 'Customer2': 2, 'Customer3': 3, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}}
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost data for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand data for customer {j}')
for i in suppliers:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity data for supplier {i}')
m = gp.Model('FreshMart_Transportation')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')