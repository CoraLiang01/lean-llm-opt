import gurobipy as gp
from gurobipy import GRB
products = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {'27in 4K Gaming Monitor': 261.2933, '27in FHD Monitor': 52.4965}
initial_inventory = {'27in 4K Gaming Monitor': 62440, '27in FHD Monitor': 75500}
demand = {'27in 4K Gaming Monitor': 12474, '27in FHD Monitor': 15057}
upper_bounds = {p: min(initial_inventory[p], demand[p]) for p in products}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('27in_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, ub=[upper_bounds[p] for p in products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] >= 0, name=f'lb_{p}')
    m.addConstr(x_vars[p] <= upper_bounds[p], name=f'ub_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')