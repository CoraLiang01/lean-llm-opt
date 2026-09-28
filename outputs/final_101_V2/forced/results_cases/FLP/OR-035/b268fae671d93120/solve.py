import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Baguette', 'Value': 888, 'Weight': 4}, {'ProductName': 'Croissant', 'Value': 134, 'Weight': 2}, {'ProductName': 'Sourdough', 'Value': 129, 'Weight': 4}, {'ProductName': 'Rye Bread', 'Value': 370, 'Weight': 3}, {'ProductName': 'Brioche', 'Value': 921, 'Weight': 2}, {'ProductName': 'Focaccia', 'Value': 765, 'Weight': 1}, {'ProductName': 'Ciabatta', 'Value': 154, 'Weight': 2}, {'ProductName': 'Pita', 'Value': 837, 'Weight': 1}, {'ProductName': 'Bagel', 'Value': 584, 'Weight': 3}, {'ProductName': 'English Muffin', 'Value': 365, 'Weight': 3}]
capacity = 180
I = [p['ProductName'] for p in products]
v = {p['ProductName']: p['Value'] for p in products}
w = {p['ProductName']: p['Weight'] for p in products}
if set(v.keys()) != set(I) or set(w.keys()) != set(I):
    raise ValueError('Missing value or weight data for some products.')
m = gp.Model('Bakery_Stocking')
x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')