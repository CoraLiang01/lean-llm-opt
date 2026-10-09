import gurobipy as gp
from gurobipy import GRB
products = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
weight = {'Spinach': 230, 'Shiitake Mushrooms': 637, 'Apples': 773, 'Carrots': 653, 'Basil': 755, 'Potatoes': 670, 'Green Beans': 505, 'Blueberries': 821, 'Oranges': 83, 'Watermelons': 249}
value = {'Spinach': 64, 'Shiitake Mushrooms': 75, 'Apples': 68, 'Carrots': 11, 'Basil': 91, 'Potatoes': 31, 'Green Beans': 90, 'Blueberries': 56, 'Oranges': 10, 'Watermelons': 24}
capacity = 875
if set(weight.keys()) != set(products) or set(value.keys()) != set(products):
    raise ValueError('Mismatch between products and weight/value keys.')
m = gp.Model('Supermarket_Stock_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')