SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 1', 'Capacity': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 2', 'Capacity': '80'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 3', 'Capacity': '120'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 4', 'Capacity': '90'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 5', 'Capacity': '50'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 6', 'Capacity': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 7', 'Capacity': '110'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 8', 'Capacity': '40'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 9', 'Capacity': '60'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', 'values': {'Warehouse ID': 'Warehouse 10', 'Capacity': '35'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
product_value = {}
product_weight = {}
warehouses = []
warehouse_capacity = {}
for rec in SOURCE_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('products.csv'):
        pname = vals['ProductName']
        products.append(pname)
        try:
            product_value[pname] = int(vals['Value'])
            product_weight[pname] = int(vals['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid product value/weight for {pname}: {e}')
    elif src.endswith('capacity.csv'):
        wid = vals['Warehouse ID']
        warehouses.append(wid)
        try:
            warehouse_capacity[wid] = int(vals['Capacity'])
        except Exception as e:
            raise ValueError(f'Invalid warehouse capacity for {wid}: {e}')
if len(products) != 10:
    raise ValueError(f'Expected 10 products, got {len(products)}')
if len(warehouses) != 10:
    raise ValueError(f'Expected 10 warehouses, got {len(warehouses)}')
if set(product_value.keys()) != set(products):
    raise ValueError('Mismatch in product values')
if set(product_weight.keys()) != set(products):
    raise ValueError('Mismatch in product weights')
if set(warehouse_capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse capacities')
m = gp.Model('NewCarSalesInNorway2')
x_vars = m.addVars(products, warehouses, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_value[i] * x_vars[i, j] for i in products for j in warehouses)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weight[i] * x_vars[i, j] for i in products)) <= warehouse_capacity[j] for j in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')