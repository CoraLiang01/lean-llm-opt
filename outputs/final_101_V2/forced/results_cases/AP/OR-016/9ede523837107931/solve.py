import gurobipy as gp
from gurobipy import GRB
products = [{'name': 'Beauty - 25', 'revenue': 25, 'initial_inventory': 1570, 'demand': 240}, {'name': 'Beauty - 30', 'revenue': 30, 'initial_inventory': 1330, 'demand': 202}, {'name': 'Beauty - 50', 'revenue': 50, 'initial_inventory': 1700, 'demand': 263}, {'name': 'Beauty - 300', 'revenue': 300, 'initial_inventory': 1420, 'demand': 216}, {'name': 'Beauty - 500', 'revenue': 500, 'initial_inventory': 1690, 'demand': 256}, {'name': 'Clothing - 25', 'revenue': 25, 'initial_inventory': 1840, 'demand': 281}, {'name': 'Clothing - 30', 'revenue': 30, 'initial_inventory': 1710, 'demand': 261}, {'name': 'Clothing - 50', 'revenue': 50, 'initial_inventory': 1890, 'demand': 290}, {'name': 'Clothing - 300', 'revenue': 300, 'initial_inventory': 1930, 'demand': 295}, {'name': 'Clothing - 500', 'revenue': 500, 'initial_inventory': 1570, 'demand': 244}, {'name': 'Electronics - 25', 'revenue': 25, 'initial_inventory': 1810, 'demand': 273}, {'name': 'Electronics - 30', 'revenue': 30, 'initial_inventory': 1410, 'demand': 220}, {'name': 'Electronics - 50', 'revenue': 50, 'initial_inventory': 1750, 'demand': 268}, {'name': 'Electronics - 300', 'revenue': 300, 'initial_inventory': 1830, 'demand': 286}, {'name': 'Electronics - 500', 'revenue': 500, 'initial_inventory': 1690, 'demand': 262}, {'name': 'Home Goods - 25', 'revenue': 25, 'initial_inventory': 1660, 'demand': 255}, {'name': 'Home Goods - 30', 'revenue': 30, 'initial_inventory': 1417, 'demand': 218}, {'name': 'Home Goods - 50', 'revenue': 50, 'initial_inventory': 1807, 'demand': 278}, {'name': 'Home Goods - 300', 'revenue': 300, 'initial_inventory': 1268, 'demand': 195}, {'name': 'Home Goods - 500', 'revenue': 500, 'initial_inventory': 1612, 'demand': 248}, {'name': 'Sports - 25', 'revenue': 25, 'initial_inventory': 1749, 'demand': 269}, {'name': 'Sports - 30', 'revenue': 30, 'initial_inventory': 1476, 'demand': 227}, {'name': 'Sports - 50', 'revenue': 50, 'initial_inventory': 1853, 'demand': 285}, {'name': 'Sports - 300', 'revenue': 300, 'initial_inventory': 1352, 'demand': 208}, {'name': 'Sports - 500', 'revenue': 500, 'initial_inventory': 1684, 'demand': 259}, {'name': 'Furniture - 25', 'revenue': 25, 'initial_inventory': 1573, 'demand': 242}, {'name': 'Furniture - 30', 'revenue': 30, 'initial_inventory': 1385, 'demand': 213}, {'name': 'Furniture - 50', 'revenue': 50, 'initial_inventory': 1768, 'demand': 272}, {'name': 'Furniture - 300', 'revenue': 300, 'initial_inventory': 1456, 'demand': 224}, {'name': 'Furniture - 500', 'revenue': 500, 'initial_inventory': 1632, 'demand': 251}, {'name': 'Toys - 25', 'revenue': 25, 'initial_inventory': 1671, 'demand': 257}, {'name': 'Toys - 30', 'revenue': 30, 'initial_inventory': 1528, 'demand': 235}, {'name': 'Toys - 50', 'revenue': 50, 'initial_inventory': 1892, 'demand': 291}, {'name': 'Toys - 300', 'revenue': 300, 'initial_inventory': 1294, 'demand': 199}, {'name': 'Toys - 500', 'revenue': 500, 'initial_inventory': 1716, 'demand': 264}]
product_names = [p['name'] for p in products]
revenue = {p['name']: p['revenue'] for p in products}
inventory = {p['name']: p['initial_inventory'] for p in products}
demand = {p['name']: p['demand'] for p in products}
for pname in product_names:
    if pname not in revenue or pname not in inventory or pname not in demand:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Retail_Fulfillment')
x = m.addVars(product_names, lb=0, ub=[min(inventory[p], demand[p]) for p in product_names], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')