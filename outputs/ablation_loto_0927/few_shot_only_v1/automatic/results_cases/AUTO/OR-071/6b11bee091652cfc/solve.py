import gurobipy as gp
from gurobipy import GRB
products = ['Classic Oxford Shirt', 'Cotton Crew Neck T-Shirt', 'French Terry Hoodie', 'Denim Snap Shirt', 'Jersey V-Neck T-Shirt', 'Unisex Jogger Pants', 'Flannel Plaid Shirt', 'Performance Polo Shirt', 'Long-sleeve Henley', 'Heavyweight Sweatshirt', 'Heavyweight Denim Shirt']
labor = {'Classic Oxford Shirt': 3.1, 'Cotton Crew Neck T-Shirt': 2.1, 'French Terry Hoodie': 6.5, 'Denim Snap Shirt': 3.4, 'Jersey V-Neck T-Shirt': 2.5, 'Unisex Jogger Pants': 5.9, 'Flannel Plaid Shirt': 3.5, 'Performance Polo Shirt': 2.9, 'Long-sleeve Henley': 2.8, 'Heavyweight Sweatshirt': 6.2, 'Heavyweight Denim Shirt': 4}
material = {'Classic Oxford Shirt': 4.2, 'Cotton Crew Neck T-Shirt': 3.1, 'French Terry Hoodie': 6.8, 'Denim Snap Shirt': 4.6, 'Jersey V-Neck T-Shirt': 3.4, 'Unisex Jogger Pants': 6.1, 'Flannel Plaid Shirt': 4, 'Performance Polo Shirt': 3.9, 'Long-sleeve Henley': 3.7, 'Heavyweight Sweatshirt': 6.4, 'Heavyweight Denim Shirt': 5.1}
selling_price = {'Classic Oxford Shirt': 125, 'Cotton Crew Neck T-Shirt': 83, 'French Terry Hoodie': 195, 'Denim Snap Shirt': 132, 'Jersey V-Neck T-Shirt': 90, 'Unisex Jogger Pants': 185, 'Flannel Plaid Shirt': 128, 'Performance Polo Shirt': 118, 'Long-sleeve Henley': 95, 'Heavyweight Sweatshirt': 190, 'Heavyweight Denim Shirt': 145}
variable_cost = {'Classic Oxford Shirt': 63, 'Cotton Crew Neck T-Shirt': 41, 'French Terry Hoodie': 95, 'Denim Snap Shirt': 68, 'Jersey V-Neck T-Shirt': 48, 'Unisex Jogger Pants': 88, 'Flannel Plaid Shirt': 65, 'Performance Polo Shirt': 57, 'Long-sleeve Henley': 50, 'Heavyweight Sweatshirt': 92, 'Heavyweight Denim Shirt': 77}
if not (len(products) == 198 and all((p in labor for p in products)) and all((p in material for p in products)) and all((p in selling_price for p in products)) and all((p in variable_cost for p in products))):
    raise ValueError('Missing data for some products.')
L = 1650
M = 1850
F = 4500
m = gp.Model('RedBeanClothingFactory')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.quicksum(((selling_price[p] - variable_cost[p]) * x[p] for p in products)) - F
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor[p] * x[p] for p in products)) <= L, name='labor')
m.addConstr(gp.quicksum((material[p] * x[p] for p in products)) <= M, name='material')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')