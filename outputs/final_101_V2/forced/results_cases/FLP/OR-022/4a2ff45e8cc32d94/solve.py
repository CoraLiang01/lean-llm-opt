import gurobipy as gp
from gurobipy import GRB
products = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {'27in 4K Gaming Monitor': 261.2933, '27in FHD Monitor': 52.4965}
inventory = {'27in 4K Gaming Monitor': 62440, '27in FHD Monitor': 75500}
demand = {'27in 4K Gaming Monitor': 12474, '27in FHD Monitor': 15057}
upper_bound = {}
for i in products:
    if i not in revenue or i not in inventory or i not in demand:
        raise ValueError(f'Missing data for product {i}')
    upper_bound[i] = min(inventory[i], demand[i])
m = gp.Model('27in_Product_Revenue_Max')
x = m.addVars(products, lb=0, ub=upper_bound, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')