import gurobipy as gp
from gurobipy import GRB
products = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
value = {'Spinach': 49, 'Shiitake Mushrooms': 30, 'Apples': 30, 'Carrots': 18, 'Basil': 54, 'Potatoes': 27, 'Green Beans': 91, 'Blueberries': 88, 'Oranges': 78, 'Watermelons': 22}
weight = {'Spinach': 282, 'Shiitake Mushrooms': 83, 'Apples': 251, 'Carrots': 257, 'Basil': 88, 'Potatoes': 52, 'Green Beans': 198, 'Blueberries': 203, 'Oranges': 87, 'Watermelons': 265}
capacity = 1035
if set(value.keys()) != set(products):
    raise ValueError('Value coefficients missing for some products.')
if set(weight.keys()) != set(products):
    raise ValueError('Weight coefficients missing for some products.')
m = gp.Model('Supermarket_Produce_Order')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')