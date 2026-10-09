import gurobipy as gp
from gurobipy import GRB
products = [{'Product Name': '27in 4K Gaming Monitor', 'Revenue': 261.2933, 'Demand': 12474, 'Initial Inventory': 62440}, {'Product Name': '27in FHD Monitor', 'Revenue': 52.4965, 'Demand': 15057, 'Initial Inventory': 75500}]
product_keys = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {'27in 4K Gaming Monitor': 261.2933, '27in FHD Monitor': 52.4965}
demand = {'27in 4K Gaming Monitor': 12474, '27in FHD Monitor': 15057}
inventory = {'27in 4K Gaming Monitor': 62440, '27in FHD Monitor': 75500}
upper_bounds = {k: min(inventory[k], demand[k]) for k in product_keys}
m = gp.Model('DeptStore_27in_Revenue')
x_vars = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, ub=[upper_bounds[k] for k in product_keys], name='')
m.setObjective(gp.quicksum((revenue[k] * x_vars[k] for k in product_keys)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')