import gurobipy as gp
from gurobipy import GRB
products = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
weights = {'Spinach': 230, 'Shiitake Mushrooms': 637, 'Apples': 773, 'Carrots': 653, 'Basil': 755, 'Potatoes': 670, 'Green Beans': 505, 'Blueberries': 821, 'Oranges': 83, 'Watermelons': 249}
values = {'Spinach': 64, 'Shiitake Mushrooms': 75, 'Apples': 68, 'Carrots': 11, 'Basil': 91, 'Potatoes': 31, 'Green Beans': 90, 'Blueberries': 56, 'Oranges': 10, 'Watermelons': 24}
capacity = 875
if set(weights.keys()) != set(products):
    raise ValueError('weights keys do not match products')
if set(values.keys()) != set(products):
    raise ValueError('values keys do not match products')
m = gp.Model('Supermarket_Stock_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')