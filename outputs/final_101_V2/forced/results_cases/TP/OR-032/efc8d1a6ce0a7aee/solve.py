import gurobipy as gp
from gurobipy import GRB
products = ['Books_15.15', 'Books_30.3', 'Books_45.45', 'Books_60.6', 'Books_75.75']
revenue = {'Books_15.15': 15.15, 'Books_30.3': 30.3, 'Books_45.45': 45.45, 'Books_60.6': 60.6, 'Books_75.75': 75.75}
demand = {'Books_15.15': 1980, 'Books_30.3': 3024, 'Books_45.45': 4536, 'Books_60.6': 5601, 'Books_75.75': 7567}
initial_inventory = {'Books_15.15': 9920.0, 'Books_30.3': 20160.0, 'Books_45.45': 30000.0, 'Books_60.6': 38360.0, 'Books_75.75': 51450.0}
for i in products:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for product {i}')
upper_bound = {i: min(demand[i], initial_inventory[i]) for i in products}
m = gp.Model('Books_Revenue_Max')
x = m.addVars(products, lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')