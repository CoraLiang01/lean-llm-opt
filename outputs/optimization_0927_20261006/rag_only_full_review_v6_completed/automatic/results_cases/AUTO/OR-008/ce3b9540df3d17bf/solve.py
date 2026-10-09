import gurobipy as gp
from gurobipy import GRB
suppliers = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
customers = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
demand = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
capacity = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
cost = {'Supplier1': {'Customer1': 2, 'Customer2': 3, 'Customer3': 1, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier2': {'Customer1': 1, 'Customer2': 2, 'Customer3': 3, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier3': {'Customer1': 3, 'Customer2': 1, 'Customer3': 2, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}, 'Supplier4': {'Customer1': 2, 'Customer2': 3, 'Customer3': 2, 'Customer4': 1, 'Customer5': 3, 'Customer6': 4}, 'Supplier5': {'Customer1': 3, 'Customer2': 2, 'Customer3': 3, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}}
for s in suppliers:
    if s not in capacity:
        raise ValueError(f'Missing capacity for supplier {s}')
    if s not in cost:
        raise ValueError(f'Missing cost row for supplier {s}')
    for c in customers:
        if c not in cost[s]:
            raise ValueError(f'Missing cost for supplier {s}, customer {c}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Missing demand for customer {c}')
m = gp.Model()
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s][c] * x_vars[s, c] for s in suppliers for c in customers)), GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c], name='')
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= capacity[s], name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for s in suppliers:
        for c in customers:
            v = x_vars[s, c]
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')