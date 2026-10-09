import gurobipy as gp
from gurobipy import GRB
products = ['Books_15.15', 'Books_30.3', 'Books_45.45', 'Books_60.6', 'Books_75.75']
revenue = {'Books_15.15': 15.15, 'Books_30.3': 30.3, 'Books_45.45': 45.45, 'Books_60.6': 60.6, 'Books_75.75': 75.75}
demand = {'Books_15.15': 1980, 'Books_30.3': 3024, 'Books_45.45': 4536, 'Books_60.6': 5601, 'Books_75.75': 7567}
initial_inventory = {'Books_15.15': 9920.0, 'Books_30.3': 20160.0, 'Books_45.45': 30000.0, 'Books_60.6': 38360.0, 'Books_75.75': 51450.0}
for p in products:
    if p not in revenue or p not in demand or p not in initial_inventory:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Books_Revenue_Max')
x_vars = m.addVars(products, lb=0, ub={p: min(demand[p], initial_inventory[p]) for p in products}, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')