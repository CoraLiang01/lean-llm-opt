import gurobipy as gp
from gurobipy import GRB
products = ['Baby Food_255.28']
revenue = {'Baby Food_255.28': 255.28}
initial_inventory = {'Baby Food_255.28': 22749210}
demand = {'Baby Food_255.28': 3066513}
for i in products:
    if i not in revenue or i not in initial_inventory or i not in demand:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Baby_Product_Revenue_Maximization')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x[i] <= initial_inventory[i], name='inv')
    m.addConstr(x[i] <= demand[i], name='dem')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')