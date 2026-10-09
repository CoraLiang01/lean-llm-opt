import gurobipy as gp
from gurobipy import GRB
products = [{'i': 1, 'ProductName': 'Spinach', 'Weight': 282, 'Value': 49}, {'i': 2, 'ProductName': 'Shiitake Mushrooms', 'Weight': 83, 'Value': 30}, {'i': 3, 'ProductName': 'Apples', 'Weight': 251, 'Value': 30}, {'i': 4, 'ProductName': 'Carrots', 'Weight': 257, 'Value': 18}, {'i': 5, 'ProductName': 'Basil', 'Weight': 88, 'Value': 54}, {'i': 6, 'ProductName': 'Potatoes', 'Weight': 52, 'Value': 27}, {'i': 7, 'ProductName': 'Green Beans', 'Weight': 198, 'Value': 91}, {'i': 8, 'ProductName': 'Blueberries', 'Weight': 203, 'Value': 88}, {'i': 9, 'ProductName': 'Oranges', 'Weight': 87, 'Value': 78}, {'i': 10, 'ProductName': 'Watermelons', 'Weight': 265, 'Value': 22}]
product_ids = [p['i'] for p in products]
product_names = {p['i']: p['ProductName'] for p in products}
weights = {p['i']: p['Weight'] for p in products}
values = {p['i']: p['Value'] for p in products}
capacity = 1035
if not set(product_ids) == set(weights.keys()) == set(values.keys()):
    raise ValueError('Mismatch in product data coverage.')
m = gp.Model('Supermarket_Produce_Order')
x_vars = m.addVars(product_ids, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x_vars[i] for i in product_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x_vars[i] for i in product_ids)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in product_ids:
        print(f'x[{i}] ({product_names[i]}): {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')