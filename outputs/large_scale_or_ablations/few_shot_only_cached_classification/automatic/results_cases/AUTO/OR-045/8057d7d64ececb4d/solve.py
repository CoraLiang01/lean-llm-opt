import gurobipy as gp
from gurobipy import GRB
produce = ['Spinach', 'Shiitake Mushrooms', 'Apples', 'Carrots', 'Basil', 'Potatoes', 'Green Beans', 'Blueberries', 'Oranges', 'Watermelons']
benefit = {'Spinach': 49, 'Shiitake Mushrooms': 30, 'Apples': 30, 'Carrots': 18, 'Basil': 54, 'Potatoes': 27, 'Green Beans': 91, 'Blueberries': 88, 'Oranges': 78, 'Watermelons': 22}
weight = {'Spinach': 282, 'Shiitake Mushrooms': 83, 'Apples': 251, 'Carrots': 257, 'Basil': 88, 'Potatoes': 52, 'Green Beans': 198, 'Blueberries': 203, 'Oranges': 87, 'Watermelons': 265}
capacity = 1035
if set(benefit.keys()) != set(produce):
    raise ValueError('Benefit data missing for some produce types.')
if set(weight.keys()) != set(produce):
    raise ValueError('Weight data missing for some produce types.')
m = gp.Model('Supermarket_Produce_Order')
x = m.addVars(produce, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[i] * x[i] for i in produce)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in produce)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')