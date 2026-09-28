import gurobipy as gp
from gurobipy import GRB
products = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
w = {'Spinach': 230, 'Shiitake Mushrooms': 637, 'Apples': 773, 'Carrots': 653, 'Basil': 755, 'Potatoes': 670, 'Green Beans': 505, 'Blueberries': 821, 'Oranges': 83, 'Watermelons': 249}
v = {'Spinach': 64, 'Shiitake Mushrooms': 75, 'Apples': 68, 'Carrots': 11, 'Basil': 91, 'Potatoes': 31, 'Green Beans': 90, 'Blueberries': 56, 'Oranges': 10, 'Watermelons': 24}
C = 875
if set(w.keys()) != set(products):
    raise ValueError('Weight data missing for some products.')
if set(v.keys()) != set(products):
    raise ValueError('Value data missing for some products.')
m = gp.Model('Supermarket_Stock_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((v[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in products)) <= C, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')