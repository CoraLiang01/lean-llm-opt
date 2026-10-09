import gurobipy as gp
from gurobipy import GRB
products = ['Classic Oxford Shirt', 'Cotton Crew Neck T-Shirt', 'French Terry Hoodie', 'Denim Snap Shirt', 'Jersey V-Neck T-Shirt', 'Unisex Jogger Pants', 'Flannel Plaid Shirt', 'Performance Polo Shirt', 'Long-sleeve Henley', 'Heavyweight Sweatshirt']
a_p = {'Classic Oxford Shirt': 3.1, 'Cotton Crew Neck T-Shirt': 2.1, 'French Terry Hoodie': 6.5, 'Denim Snap Shirt': 3.4, 'Jersey V-Neck T-Shirt': 2.5, 'Unisex Jogger Pants': 5.9, 'Flannel Plaid Shirt': 3.5, 'Performance Polo Shirt': 2.9, 'Long-sleeve Henley': 2.8, 'Heavyweight Sweatshirt': 6.2}
b_p = {'Classic Oxford Shirt': 4.2, 'Cotton Crew Neck T-Shirt': 3.1, 'French Terry Hoodie': 6.8, 'Denim Snap Shirt': 4.6, 'Jersey V-Neck T-Shirt': 3.4, 'Unisex Jogger Pants': 6.1, 'Flannel Plaid Shirt': 4.0, 'Performance Polo Shirt': 3.9, 'Long-sleeve Henley': 3.7, 'Heavyweight Sweatshirt': 6.4}
s_p = {'Classic Oxford Shirt': 125, 'Cotton Crew Neck T-Shirt': 83, 'French Terry Hoodie': 195, 'Denim Snap Shirt': 132, 'Jersey V-Neck T-Shirt': 90, 'Unisex Jogger Pants': 185, 'Flannel Plaid Shirt': 128, 'Performance Polo Shirt': 118, 'Long-sleeve Henley': 95, 'Heavyweight Sweatshirt': 190}
v_p = {'Classic Oxford Shirt': 63, 'Cotton Crew Neck T-Shirt': 41, 'French Terry Hoodie': 95, 'Denim Snap Shirt': 68, 'Jersey V-Neck T-Shirt': 48, 'Unisex Jogger Pants': 88, 'Flannel Plaid Shirt': 65, 'Performance Polo Shirt': 57, 'Long-sleeve Henley': 50, 'Heavyweight Sweatshirt': 92}
for p in products:
    if p not in a_p or p not in b_p or p not in s_p or (p not in v_p):
        raise ValueError(f'Missing data for product: {p}')
L = 1650
M = 1850
F = 4500
m = gp.Model('RedBeanClothingFactory')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.quicksum(((s_p[p] - v_p[p]) * x_vars[p] for p in products)) - F
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((a_p[p] * x_vars[p] for p in products)) <= L, name='labor')
m.addConstr(gp.quicksum((b_p[p] * x_vars[p] for p in products)) <= M, name='material')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')