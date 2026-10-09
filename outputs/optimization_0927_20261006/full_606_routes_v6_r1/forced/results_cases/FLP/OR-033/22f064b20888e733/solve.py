import gurobipy as gp
from gurobipy import GRB
products = ['Baby Food_255.28']
revenue = {'Baby Food_255.28': 255.28}
initial_inventory = {'Baby Food_255.28': 5627060}
demand = {'Baby Food_255.28': 765850}
for i in products:
    if i not in revenue or i not in initial_inventory or i not in demand:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, ub={i: min(initial_inventory[i], demand[i]) for i in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')