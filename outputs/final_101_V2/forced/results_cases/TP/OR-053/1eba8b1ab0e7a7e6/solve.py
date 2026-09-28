import gurobipy as gp
from gurobipy import GRB
shelves = [f'Shelf{i}' for i in range(1, 11)]
products = [f'Product{j}' for j in range(1, 21)]
capacity = {'Shelf1': 500, 'Shelf2': 700, 'Shelf3': 600, 'Shelf4': 800, 'Shelf5': 550, 'Shelf6': 900, 'Shelf7': 650, 'Shelf8': 750, 'Shelf9': 820, 'Shelf10': 570}
product_value = {'Product1': 50, 'Product2': 70, 'Product3': 30, 'Product4': 60, 'Product5': 80, 'Product6': 90, 'Product7': 40, 'Product8': 100, 'Product9': 55, 'Product10': 75, 'Product11': 65, 'Product12': 95, 'Product13': 45, 'Product14': 85, 'Product15': 70, 'Product16': 110, 'Product17': 50, 'Product18': 60, 'Product19': 120, 'Product20': 100}
product_weight = {'Product1': 10, 'Product2': 20, 'Product3': 5, 'Product4': 15, 'Product5': 25, 'Product6': 30, 'Product7': 12, 'Product8': 35, 'Product9': 10, 'Product10': 20, 'Product11': 18, 'Product12': 28, 'Product13': 8, 'Product14': 22, 'Product15': 25, 'Product16': 40, 'Product17': 14, 'Product18': 16, 'Product19': 50, 'Product20': 30}
if set(capacity.keys()) != set(shelves):
    raise ValueError('Shelf capacity data missing or extra entries.')
if set(product_value.keys()) != set(products):
    raise ValueError('Product value data missing or extra entries.')
if set(product_weight.keys()) != set(products):
    raise ValueError('Product weight data missing or extra entries.')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weight[p] * x[s, p] for p in products)) <= capacity[s] for s in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')