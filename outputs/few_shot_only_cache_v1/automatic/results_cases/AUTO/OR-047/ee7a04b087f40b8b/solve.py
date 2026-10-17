SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '1', 'Capacity': '1336'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '2', 'Capacity': '1754'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '3', 'Capacity': '1617'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '4', 'Capacity': '1119'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '5', 'Capacity': '1410'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '6', 'Capacity': '627'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '7', 'Capacity': '748'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '8', 'Capacity': '1540'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '9', 'Capacity': '1292'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'values': {'PlatformId': '10', 'Capacity': '1138'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Racing', 'Value': '28', 'Weight': '393'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Sports', 'Value': '69', 'Weight': '195'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Action', 'Value': '20', 'Weight': '192'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Adventure', 'Value': '62', 'Weight': '155'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'RPG', 'Value': '58', 'Weight': '500'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Shooter', 'Value': '11', 'Weight': '156'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Strategy', 'Value': '73', 'Weight': '317'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Simulation', 'Value': '43', 'Weight': '694'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Puzzle', 'Value': '28', 'Weight': '751'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Fighting', 'Value': '57', 'Weight': '467'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Platformer', 'Value': '92', 'Weight': '796'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Survival', 'Value': '66', 'Weight': '146'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '269'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'Sandbox', 'Value': '49', 'Weight': '246'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', 'values': {'ProductName': 'MMO', 'Value': '12', 'Weight': '652'}}]
import gurobipy as gp
from gurobipy import GRB
platform_caps = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('capacity.csv'):
        pid = rec['values']['PlatformId']
        cap = int(rec['values']['Capacity'])
        platform_caps[pid] = cap
product_values = {}
product_weights = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('products.csv'):
        pname = rec['values']['ProductName']
        product_values[pname] = int(rec['values']['Value'])
        product_weights[pname] = int(rec['values']['Weight'])
if not platform_caps or not product_values or (not product_weights):
    raise ValueError('Missing platform or product data in SOURCE_RECORDS.')
if set(product_values) != set(product_weights):
    raise ValueError('Mismatch in product names between values and weights.')
platforms = sorted(platform_caps.keys(), key=lambda x: int(x))
products = list(product_values.keys())
m = gp.Model('VideoGameStore_Allocation')
x_vars = m.addVars(platforms, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in platforms for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weights[j] * x_vars[i, j] for j in products)) <= platform_caps[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')