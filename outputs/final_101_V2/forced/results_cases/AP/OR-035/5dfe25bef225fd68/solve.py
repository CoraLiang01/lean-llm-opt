import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Baguette', 'Value': 888, 'Weight': 4}, {'ProductName': 'Croissant', 'Value': 134, 'Weight': 2}, {'ProductName': 'Sourdough', 'Value': 129, 'Weight': 4}, {'ProductName': 'Rye Bread', 'Value': 370, 'Weight': 3}, {'ProductName': 'Brioche', 'Value': 921, 'Weight': 2}, {'ProductName': 'Focaccia', 'Value': 765, 'Weight': 1}, {'ProductName': 'Ciabatta', 'Value': 154, 'Weight': 2}, {'ProductName': 'Pita', 'Value': 837, 'Weight': 1}, {'ProductName': 'Bagel', 'Value': 584, 'Weight': 3}, {'ProductName': 'English Muffin', 'Value': 365, 'Weight': 3}]
capacity = 180
product_indices = list(range(1, 11))
product_names = {i: products[i - 1]['ProductName'] for i in product_indices}
values = {i: products[i - 1]['Value'] for i in product_indices}
weights = {i: products[i - 1]['Weight'] for i in product_indices}
if not (set(values.keys()) == set(product_indices) and set(weights.keys()) == set(product_indices)):
    raise ValueError('Missing value or weight data for some products.')
m = gp.Model('Bakery_Stocking')
x = m.addVars(product_indices, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in product_indices)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in product_indices)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in product_indices:
        print(f'x[{i}] ({product_names[i]}): {x[i].X}')
else:
    print(f'Solver status: {m.Status}')