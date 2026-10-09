import gurobipy as gp
from gurobipy import GRB
products = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
profit = {'Baguette': 888, 'Croissant': 134, 'Sourdough': 129, 'Rye Bread': 370, 'Brioche': 921, 'Focaccia': 765, 'Ciabatta': 154, 'Pita': 837, 'Bagel': 584, 'English Muffin': 365}
weight = {'Baguette': 4, 'Croissant': 2, 'Sourdough': 4, 'Rye Bread': 3, 'Brioche': 2, 'Focaccia': 1, 'Ciabatta': 2, 'Pita': 1, 'Bagel': 3, 'English Muffin': 3}
capacity = 180
if set(profit.keys()) != set(products):
    raise ValueError('Profit data missing for some products.')
if set(weight.keys()) != set(products):
    raise ValueError('Weight data missing for some products.')
m = gp.Model('Bakery_Bread_Stocking')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((profit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='storage')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')