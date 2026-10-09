import gurobipy as gp
from gurobipy import GRB
bread_types = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
profits = {'Baguette': 888, 'Croissant': 134, 'Sourdough': 129, 'Rye Bread': 370, 'Brioche': 921, 'Focaccia': 765, 'Ciabatta': 154, 'Pita': 837, 'Bagel': 584, 'English Muffin': 365}
weights = {'Baguette': 4, 'Croissant': 2, 'Sourdough': 4, 'Rye Bread': 3, 'Brioche': 2, 'Focaccia': 1, 'Ciabatta': 2, 'Pita': 1, 'Bagel': 3, 'English Muffin': 3}
storage_capacity = 180
if set(bread_types) != set(profits.keys()) or set(bread_types) != set(weights.keys()):
    raise ValueError('Mismatch in bread_types, profits, or weights keys.')
m = gp.Model('Bakery_Bread_Order')
x_vars = m.addVars(bread_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profits[b] * x_vars[b] for b in bread_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[b] * x_vars[b] for b in bread_types)) <= storage_capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')