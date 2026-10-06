import gurobipy as gp
from gurobipy import GRB
products = ['Beauty - 25', 'Beauty - 30', 'Beauty - 300', 'Beauty - 50', 'Beauty - 500', 'Clothing - 25', 'Clothing - 30', 'Clothing - 300', 'Clothing - 50', 'Clothing - 500', 'Electronics - 25', 'Electronics - 30', 'Electronics - 300', 'Electronics - 50', 'Electronics - 500', 'Home Goods - 25', 'Home Goods - 30', 'Home Goods - 50', 'Home Goods - 300', 'Home Goods - 500', 'Sports - 25', 'Sports - 30', 'Sports - 50', 'Sports - 300', 'Sports - 500', 'Furniture - 25', 'Furniture - 30', 'Furniture - 50', 'Furniture - 300', 'Furniture - 500', 'Toys - 25', 'Toys - 30', 'Toys - 50', 'Toys - 300', 'Toys - 500']
revenue = {'Beauty - 25': 25, 'Beauty - 30': 30, 'Beauty - 300': 300, 'Beauty - 50': 50, 'Beauty - 500': 500, 'Clothing - 25': 25, 'Clothing - 30': 30, 'Clothing - 300': 300, 'Clothing - 50': 50, 'Clothing - 500': 500, 'Electronics - 25': 25, 'Electronics - 30': 30, 'Electronics - 300': 300, 'Electronics - 50': 50, 'Electronics - 500': 500, 'Home Goods - 25': 25, 'Home Goods - 30': 30, 'Home Goods - 50': 50, 'Home Goods - 300': 300, 'Home Goods - 500': 500, 'Sports - 25': 25, 'Sports - 30': 30, 'Sports - 50': 50, 'Sports - 300': 300, 'Sports - 500': 500, 'Furniture - 25': 25, 'Furniture - 30': 30, 'Furniture - 50': 50, 'Furniture - 300': 300, 'Furniture - 500': 500, 'Toys - 25': 25, 'Toys - 30': 30, 'Toys - 50': 50, 'Toys - 300': 300, 'Toys - 500': 500}
demand = {'Beauty - 25': 240, 'Beauty - 30': 202, 'Beauty - 300': 216, 'Beauty - 50': 263, 'Beauty - 500': 256, 'Clothing - 25': 281, 'Clothing - 30': 261, 'Clothing - 300': 295, 'Clothing - 50': 290, 'Clothing - 500': 244, 'Electronics - 25': 273, 'Electronics - 30': 220, 'Electronics - 300': 286, 'Electronics - 50': 268, 'Electronics - 500': 262, 'Home Goods - 25': 255, 'Home Goods - 30': 218, 'Home Goods - 50': 278, 'Home Goods - 300': 195, 'Home Goods - 500': 248, 'Sports - 25': 269, 'Sports - 30': 227, 'Sports - 50': 285, 'Sports - 300': 208, 'Sports - 500': 259, 'Furniture - 25': 242, 'Furniture - 30': 213, 'Furniture - 50': 272, 'Furniture - 300': 224, 'Furniture - 500': 251, 'Toys - 25': 257, 'Toys - 30': 235, 'Toys - 50': 291, 'Toys - 300': 199, 'Toys - 500': 264}
inventory = {'Beauty - 25': 1570, 'Beauty - 30': 1330, 'Beauty - 300': 1420, 'Beauty - 50': 1700, 'Beauty - 500': 1690, 'Clothing - 25': 1840, 'Clothing - 30': 1710, 'Clothing - 300': 1930, 'Clothing - 50': 1890, 'Clothing - 500': 1570, 'Electronics - 25': 1810, 'Electronics - 30': 1410, 'Electronics - 300': 1830, 'Electronics - 50': 1750, 'Electronics - 500': 1690, 'Home Goods - 25': 1660, 'Home Goods - 30': 1417, 'Home Goods - 50': 1807, 'Home Goods - 300': 1268, 'Home Goods - 500': 1612, 'Sports - 25': 1749, 'Sports - 30': 1476, 'Sports - 50': 1853, 'Sports - 300': 1352, 'Sports - 500': 1684, 'Furniture - 25': 1573, 'Furniture - 30': 1385, 'Furniture - 50': 1768, 'Furniture - 300': 1456, 'Furniture - 500': 1632, 'Toys - 25': 1671, 'Toys - 30': 1528, 'Toys - 50': 1892, 'Toys - 300': 1294, 'Toys - 500': 1716}
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('Retail_Revenue_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= inventory[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')