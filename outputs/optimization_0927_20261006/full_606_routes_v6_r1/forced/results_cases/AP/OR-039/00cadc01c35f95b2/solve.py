import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Sedans', 'Value': 1200, 'Weight': 20}, {'ProductName': 'SUVs', 'Value': 1800, 'Weight': 15}, {'ProductName': 'Electric Vehicles', 'Value': 2500, 'Weight': 25}, {'ProductName': 'Hybrid Vehicles', 'Value': 2000, 'Weight': 18}, {'ProductName': 'Trucks', 'Value': 1500, 'Weight': 10}, {'ProductName': 'Sports Cars', 'Value': 3000, 'Weight': 5}, {'ProductName': 'Compact Cars', 'Value': 1000, 'Weight': 22}, {'ProductName': 'Luxury Sedans', 'Value': 3500, 'Weight': 8}, {'ProductName': 'Vans', 'Value': 1600, 'Weight': 12}, {'ProductName': 'Pickup Trucks', 'Value': 1700, 'Weight': 7}]
warehouses = [{'Warehouse ID': 'Warehouse 1', 'Capacity': 100}, {'Warehouse ID': 'Warehouse 2', 'Capacity': 80}, {'Warehouse ID': 'Warehouse 3', 'Capacity': 120}, {'Warehouse ID': 'Warehouse 4', 'Capacity': 90}, {'Warehouse ID': 'Warehouse 5', 'Capacity': 50}, {'Warehouse ID': 'Warehouse 6', 'Capacity': 30}, {'Warehouse ID': 'Warehouse 7', 'Capacity': 110}, {'Warehouse ID': 'Warehouse 8', 'Capacity': 40}, {'Warehouse ID': 'Warehouse 9', 'Capacity': 60}, {'Warehouse ID': 'Warehouse 10', 'Capacity': 35}]
product_ids = [p['ProductName'] for p in products]
warehouse_ids = [w['Warehouse ID'] for w in warehouses]
v = {p['ProductName']: p['Value'] for p in products}
w = {p['ProductName']: p['Weight'] for p in products}
C = {w['Warehouse ID']: w['Capacity'] for w in warehouses}
if set(v.keys()) != set(product_ids) or set(w.keys()) != set(product_ids):
    raise ValueError('Product value/weight keys do not match product_ids')
if set(C.keys()) != set(warehouse_ids):
    raise ValueError('Warehouse capacity keys do not match warehouse_ids')
m = gp.Model('Car_Inventory_Optimization')
x_vars = m.addVars(product_ids, warehouse_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[i] * x_vars[i, j] for i in product_ids for j in warehouse_ids)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[i] * x_vars[i, j] for i in product_ids)) <= C[j] for j in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')