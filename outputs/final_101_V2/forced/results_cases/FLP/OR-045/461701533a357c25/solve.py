import gurobipy as gp
from gurobipy import GRB
produce = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
weights = {'Spinach': 282, 'Shiitake Mushrooms': 83, 'Apples': 251, 'Carrots': 257, 'Basil': 88, 'Potatoes': 52, 'Green Beans': 198, 'Blueberries': 203, 'Oranges': 87, 'Watermelons': 265}
values = {'Spinach': 49, 'Shiitake Mushrooms': 30, 'Apples': 30, 'Carrots': 18, 'Basil': 54, 'Potatoes': 27, 'Green Beans': 91, 'Blueberries': 88, 'Oranges': 78, 'Watermelons': 22}
capacity = 1035
if set(weights.keys()) != set(produce):
    raise ValueError('weights keys do not match produce set')
if set(values.keys()) != set(produce):
    raise ValueError('values keys do not match produce set')
m = gp.Model('Supermarket_Produce_Order')
x = m.addVars(produce, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in produce)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in produce)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')