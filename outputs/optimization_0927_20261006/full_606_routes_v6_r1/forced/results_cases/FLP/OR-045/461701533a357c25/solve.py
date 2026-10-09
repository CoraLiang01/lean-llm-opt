import gurobipy as gp
from gurobipy import GRB
products = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
weights = {'Spinach': 282, 'Shiitake Mushrooms': 83, 'Apples': 251, 'Carrots': 257, 'Basil': 88, 'Potatoes': 52, 'Green Beans': 198, 'Blueberries': 203, 'Oranges': 87, 'Watermelons': 265}
values = {'Spinach': 49, 'Shiitake Mushrooms': 30, 'Apples': 30, 'Carrots': 18, 'Basil': 54, 'Potatoes': 27, 'Green Beans': 91, 'Blueberries': 88, 'Oranges': 78, 'Watermelons': 22}
capacity = 1035
if set(weights.keys()) != set(products):
    raise ValueError('weights keys do not match products')
if set(values.keys()) != set(products):
    raise ValueError('values keys do not match products')

def build_model():
    m = gp.Model('Supermarket_Produce_Order')
    x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= capacity, name='cap')
    m.Params.MIPGap = 0.0001
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')