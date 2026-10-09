import gurobipy as gp
from gurobipy import GRB
bread_types = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
profit = {'Baguette': 888, 'Croissant': 134, 'Sourdough': 129, 'Rye Bread': 370, 'Brioche': 921, 'Focaccia': 765, 'Ciabatta': 154, 'Pita': 837, 'Bagel': 584, 'English Muffin': 365}
weight = {'Baguette': 4, 'Croissant': 2, 'Sourdough': 4, 'Rye Bread': 3, 'Brioche': 2, 'Focaccia': 1, 'Ciabatta': 2, 'Pita': 1, 'Bagel': 3, 'English Muffin': 3}
capacity = 180
if set(profit.keys()) != set(bread_types):
    raise ValueError('Profit data missing for some bread types.')
if set(weight.keys()) != set(bread_types):
    raise ValueError('Weight data missing for some bread types.')
m = gp.Model('Bakery_Bread_Stocking')
x_vars = m.addVars(bread_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x_vars[i] for i in bread_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in bread_types)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')