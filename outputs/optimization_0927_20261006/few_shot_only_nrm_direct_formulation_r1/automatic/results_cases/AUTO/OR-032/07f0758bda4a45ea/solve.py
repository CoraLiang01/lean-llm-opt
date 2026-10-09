import gurobipy as gp
from gurobipy import GRB
books = ['Books_15.15', 'Books_30.3', 'Books_45.45', 'Books_60.6', 'Books_75.75']
revenue = {'Books_15.15': 15.15, 'Books_30.3': 30.3, 'Books_45.45': 45.45, 'Books_60.6': 60.6, 'Books_75.75': 75.75}
demand = {'Books_15.15': 1980, 'Books_30.3': 3024, 'Books_45.45': 4536, 'Books_60.6': 5601, 'Books_75.75': 7567}
inventory = {'Books_15.15': 9920.0, 'Books_30.3': 20160.0, 'Books_45.45': 30000.0, 'Books_60.6': 38360.0, 'Books_75.75': 51450.0}
for k in books:
    if k not in revenue or k not in demand or k not in inventory:
        raise ValueError(f'Missing data for {k}')
m = gp.Model('Books_Revenue_Maximization')
x_vars = m.addVars(books, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[k] * x_vars[k] for k in books)), GRB.MAXIMIZE)
m.addConstrs((x_vars[k] <= inventory[k] for k in books), name='')
m.addConstrs((x_vars[k] <= demand[k] for k in books), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')