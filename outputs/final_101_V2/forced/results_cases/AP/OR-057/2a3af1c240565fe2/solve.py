import gurobipy as gp
from gurobipy import GRB
platforms = {1: 995, 2: 1143, 3: 949, 4: 969, 5: 1649, 6: 870, 7: 1064, 8: 536, 9: 766, 10: 532, 11: 1703, 12: 1633, 13: 1203, 14: 1979, 15: 1797}
products = [{'ProductName': 'Racing', 'Value': 59, 'Weight': 776}, {'ProductName': 'Sports', 'Value': 83, 'Weight': 573}, {'ProductName': 'Action', 'Value': 94, 'Weight': 127}, {'ProductName': 'Adventure', 'Value': 41, 'Weight': 138}, {'ProductName': 'RPG', 'Value': 96, 'Weight': 385}, {'ProductName': 'Shooter', 'Value': 12, 'Weight': 263}, {'ProductName': 'Strategy', 'Value': 83, 'Weight': 473}, {'ProductName': 'Simulation', 'Value': 36, 'Weight': 387}, {'ProductName': 'Puzzle', 'Value': 56, 'Weight': 390}, {'ProductName': 'Fighting', 'Value': 27, 'Weight': 556}, {'ProductName': 'Platformer', 'Value': 47, 'Weight': 601}, {'ProductName': 'Survival', 'Value': 24, 'Weight': 441}, {'ProductName': 'Horror', 'Value': 14, 'Weight': 603}, {'ProductName': 'Sandbox', 'Value': 22, 'Weight': 411}, {'ProductName': 'MMO', 'Value': 17, 'Weight': 652}]
platform_ids = list(platforms.keys())
product_ids = list(range(1, 16))
v = {j + 1: products[j]['Value'] for j in range(15)}
w = {j + 1: products[j]['Weight'] for j in range(15)}
if set(platform_ids) != set(range(1, 16)):
    raise ValueError('Platform IDs must be 1..15')
if set(product_ids) != set(range(1, 16)):
    raise ValueError('Product IDs must be 1..15')
if any((j not in v or j not in w for j in product_ids)):
    raise ValueError('Missing value or weight for some product IDs')
m = gp.Model('Game_Platform_Listing')
x = m.addVars(platform_ids, product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x[i, j] for i in platform_ids for j in product_ids)), GRB.MAXIMIZE)
for i in platform_ids:
    m.addConstr(gp.quicksum((w[j] * x[i, j] for j in product_ids)) <= platforms[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')