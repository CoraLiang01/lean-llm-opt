import gurobipy as gp
from gurobipy import GRB
products = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
profit = {'Baguette': 888, 'Croissant': 134, 'Sourdough': 129, 'Rye Bread': 370, 'Brioche': 921, 'Focaccia': 765, 'Ciabatta': 154, 'Pita': 837, 'Bagel': 584, 'English Muffin': 365}
weight = {'Baguette': 4, 'Croissant': 2, 'Sourdough': 4, 'Rye Bread': 3, 'Brioche': 2, 'Focaccia': 1, 'Ciabatta': 2, 'Pita': 1, 'Bagel': 3, 'English Muffin': 3}
storage_capacity = 180
if set(profit.keys()) != set(products):
    raise ValueError('Profit data missing for some products.')
if set(weight.keys()) != set(products):
    raise ValueError('Weight data missing for some products.')
m = gp.Model('Bakery_Stocking')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= storage_capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')