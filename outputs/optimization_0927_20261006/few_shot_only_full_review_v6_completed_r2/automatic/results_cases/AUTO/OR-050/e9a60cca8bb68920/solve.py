import gurobipy as gp
from gurobipy import GRB
displays = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
products = ['Smartphone', 'Laptop', 'Headphones', 'Camera', 'Smartwatch', 'Tablet', 'Bluetooth Speaker', 'Keyboard', 'Mouse', 'Monitor', 'Printer', 'External Hard Drive', 'Router', 'Power Bank', 'Memory Card', 'USB Flash Drive', 'Smart Home Hub', 'Gaming Console', 'Fitness Tracker', 'E-Reader']
capacity = {1: 5.0, 2: 7.0, 3: 6.0, 4: 8.0, 5: 5.5, 6: 9.0, 7: 6.5, 8: 7.5, 9: 8.2, 10: 5.7}
value = {'Smartphone': 200, 'Laptop': 1500, 'Headphones': 100, 'Camera': 800, 'Smartwatch': 250, 'Tablet': 600, 'Bluetooth Speaker': 150, 'Keyboard': 80, 'Mouse': 50, 'Monitor': 300, 'Printer': 400, 'External Hard Drive': 120, 'Router': 60, 'Power Bank': 40, 'Memory Card': 30, 'USB Flash Drive': 25, 'Smart Home Hub': 100, 'Gaming Console': 500, 'Fitness Tracker': 90, 'E-Reader': 180}
weight = {'Smartphone': 1.0, 'Laptop': 5.0, 'Headphones': 0.5, 'Camera': 2.0, 'Smartwatch': 0.3, 'Tablet': 1.5, 'Bluetooth Speaker': 1.0, 'Keyboard': 0.8, 'Mouse': 0.2, 'Monitor': 3.0, 'Printer': 4.0, 'External Hard Drive': 0.5, 'Router': 0.3, 'Power Bank': 0.4, 'Memory Card': 0.05, 'USB Flash Drive': 0.02, 'Smart Home Hub': 0.6, 'Gaming Console': 4.0, 'Fitness Tracker': 0.2, 'E-Reader': 0.5}
for i in displays:
    if i not in capacity:
        raise ValueError(f'Missing capacity for display {i}')
for j in products:
    if j not in value:
        raise ValueError(f'Missing value for product {j}')
    if j not in weight:
        raise ValueError(f'Missing weight for product {j}')
m = gp.Model('Retail_Display_Allocation')
x_vars = m.addVars(displays, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in displays for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in displays), name='')
m.addConstr(gp.quicksum((x_vars[i, 'Smartphone'] for i in displays)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')