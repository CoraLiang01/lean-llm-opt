import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Baguette', 'Value': 888, 'Weight': 4}, {'ProductName': 'Croissant', 'Value': 134, 'Weight': 2}, {'ProductName': 'Sourdough', 'Value': 129, 'Weight': 4}, {'ProductName': 'Rye Bread', 'Value': 370, 'Weight': 3}, {'ProductName': 'Brioche', 'Value': 921, 'Weight': 2}, {'ProductName': 'Focaccia', 'Value': 765, 'Weight': 1}, {'ProductName': 'Ciabatta', 'Value': 154, 'Weight': 2}, {'ProductName': 'Pita', 'Value': 837, 'Weight': 1}, {'ProductName': 'Bagel', 'Value': 584, 'Weight': 3}, {'ProductName': 'English Muffin', 'Value': 365, 'Weight': 3}]
capacity = 180
product_names = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
if set(values.keys()) != set(product_names) or set(weights.keys()) != set(product_names):
    raise ValueError('Missing value or weight data for some products.')
m = gp.Model('Bakery_Stocking')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')