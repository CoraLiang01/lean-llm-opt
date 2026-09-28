import gurobipy as gp
from gurobipy import GRB
storage_areas = [{'StorageID': '1', 'Capacity': 1083}, {'StorageID': '2', 'Capacity': 1840}, {'StorageID': '3', 'Capacity': 770}, {'StorageID': '4', 'Capacity': 1299}, {'StorageID': '5', 'Capacity': 1259}, {'StorageID': '6', 'Capacity': 543}, {'StorageID': '7', 'Capacity': 1831}, {'StorageID': '8', 'Capacity': 855}, {'StorageID': '9', 'Capacity': 619}, {'StorageID': '10', 'Capacity': 637}, {'StorageID': '11', 'Capacity': 935}, {'StorageID': '12', 'Capacity': 626}, {'StorageID': '13', 'Capacity': 1457}, {'StorageID': '14', 'Capacity': 1198}, {'StorageID': '15', 'Capacity': 837}]
products = [{'ProductName': 'Window Unit', 'Value': 4811, 'Weight': 114}, {'ProductName': 'Portable Unit', 'Value': 1130, 'Weight': 200}, {'ProductName': 'Split System', 'Value': 1611, 'Weight': 106}, {'ProductName': 'Ductless System', 'Value': 3368, 'Weight': 256}, {'ProductName': 'Central AC', 'Value': 2135, 'Weight': 268}, {'ProductName': 'Hybrid AC', 'Value': 1046, 'Weight': 185}, {'ProductName': 'Geothermal AC', 'Value': 4030, 'Weight': 299}, {'ProductName': 'Smart AC', 'Value': 3761, 'Weight': 131}, {'ProductName': 'Evaporative Cooler', 'Value': 3523, 'Weight': 139}, {'ProductName': 'Package Unit', 'Value': 1701, 'Weight': 105}]
storage_ids = [area['StorageID'] for area in storage_areas]
product_names = [prod['ProductName'] for prod in products]
C = {area['StorageID']: area['Capacity'] for area in storage_areas}
v = {prod['ProductName']: prod['Value'] for prod in products}
w = {prod['ProductName']: prod['Weight'] for prod in products}
if len(storage_ids) != 15 or len(product_names) != 10:
    raise ValueError('Storage area or product count mismatch.')
if set(C.keys()) != set(storage_ids):
    raise ValueError('Missing storage area capacities.')
if set(v.keys()) != set(product_names) or set(w.keys()) != set(product_names):
    raise ValueError('Missing product values or weights.')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x[i, j] for i in storage_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in product_names)) <= C[i] for i in storage_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')