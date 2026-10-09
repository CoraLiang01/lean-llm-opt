import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Spinach', 'Weight': 230, 'Value': 64}, {'ProductName': 'Shiitake Mushrooms', 'Weight': 637, 'Value': 75}, {'ProductName': 'Apples', 'Weight': 773, 'Value': 68}, {'ProductName': 'Carrots', 'Weight': 653, 'Value': 11}, {'ProductName': 'Basil', 'Weight': 755, 'Value': 91}, {'ProductName': 'Potatoes', 'Weight': 670, 'Value': 31}, {'ProductName': 'Green Beans', 'Weight': 505, 'Value': 90}, {'ProductName': 'Blueberries', 'Weight': 821, 'Value': 56}, {'ProductName': 'Oranges', 'Weight': 83, 'Value': 10}, {'ProductName': 'Watermelons', 'Weight': 249, 'Value': 24}]
capacity = 875
product_names = [p['ProductName'] for p in products]
weights = {p['ProductName']: p['Weight'] for p in products}
values = {p['ProductName']: p['Value'] for p in products}
m = gp.Model('Supermarket_Stock_Optimization')
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[name] * x_vars[name] for name in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[name] * x_vars[name] for name in product_names)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')