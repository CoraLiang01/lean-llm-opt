LEGACY_OBSERVATION = '{"values": {"StorageID": "1", "Capacity": "1083"}}\n{"values": {"StorageID": "2", "Capacity": "1840"}}\n{"values": {"StorageID": "3", "Capacity": "770"}}\n{"values": {"StorageID": "4", "Capacity": "1299"}}\n{"values": {"StorageID": "5", "Capacity": "1259"}}\n{"values": {"StorageID": "6", "Capacity": "543"}}\n{"values": {"StorageID": "7", "Capacity": "1831"}}\n{"values": {"StorageID": "8", "Capacity": "855"}}\n{"values": {"StorageID": "9", "Capacity": "619"}}\n{"values": {"StorageID": "10", "Capacity": "637"}}\n{"values": {"StorageID": "11", "Capacity": "935"}}\n{"values": {"StorageID": "12", "Capacity": "626"}}\n{"values": {"StorageID": "13", "Capacity": "1457"}}\n{"values": {"StorageID": "14", "Capacity": "1198"}}\n{"values": {"StorageID": "15", "Capacity": "837"}}\n{"values": {"ProductName": "Window Unit", "Value": "4811", "Weight": "114"}}\n{"values": {"ProductName": "Portable Unit", "Value": "1130", "Weight": "200"}}\n{"values": {"ProductName": "Split System", "Value": "1611", "Weight": "106"}}\n{"values": {"ProductName": "Ductless System", "Value": "3368", "Weight": "256"}}\n{"values": {"ProductName": "Central AC", "Value": "2135", "Weight": "268"}}\n{"values": {"ProductName": "Hybrid AC", "Value": "1046", "Weight": "185"}}\n{"values": {"ProductName": "Geothermal AC", "Value": "4030", "Weight": "299"}}\n{"values": {"ProductName": "Smart AC", "Value": "3761", "Weight": "131"}}\n{"values": {"ProductName": "Evaporative Cooler", "Value": "3523", "Weight": "139"}}\n{"values": {"ProductName": "Package Unit", "Value": "1701", "Weight": "105"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'StorageID': '1', 'Capacity': '1083'}}, {'source': '', 'values': {'StorageID': '2', 'Capacity': '1840'}}, {'source': '', 'values': {'StorageID': '3', 'Capacity': '770'}}, {'source': '', 'values': {'StorageID': '4', 'Capacity': '1299'}}, {'source': '', 'values': {'StorageID': '5', 'Capacity': '1259'}}, {'source': '', 'values': {'StorageID': '6', 'Capacity': '543'}}, {'source': '', 'values': {'StorageID': '7', 'Capacity': '1831'}}, {'source': '', 'values': {'StorageID': '8', 'Capacity': '855'}}, {'source': '', 'values': {'StorageID': '9', 'Capacity': '619'}}, {'source': '', 'values': {'StorageID': '10', 'Capacity': '637'}}, {'source': '', 'values': {'StorageID': '11', 'Capacity': '935'}}, {'source': '', 'values': {'StorageID': '12', 'Capacity': '626'}}, {'source': '', 'values': {'StorageID': '13', 'Capacity': '1457'}}, {'source': '', 'values': {'StorageID': '14', 'Capacity': '1198'}}, {'source': '', 'values': {'StorageID': '15', 'Capacity': '837'}}, {'source': '', 'values': {'ProductName': 'Window Unit', 'Value': '4811', 'Weight': '114'}}, {'source': '', 'values': {'ProductName': 'Portable Unit', 'Value': '1130', 'Weight': '200'}}, {'source': '', 'values': {'ProductName': 'Split System', 'Value': '1611', 'Weight': '106'}}, {'source': '', 'values': {'ProductName': 'Ductless System', 'Value': '3368', 'Weight': '256'}}, {'source': '', 'values': {'ProductName': 'Central AC', 'Value': '2135', 'Weight': '268'}}, {'source': '', 'values': {'ProductName': 'Hybrid AC', 'Value': '1046', 'Weight': '185'}}, {'source': '', 'values': {'ProductName': 'Geothermal AC', 'Value': '4030', 'Weight': '299'}}, {'source': '', 'values': {'ProductName': 'Smart AC', 'Value': '3761', 'Weight': '131'}}, {'source': '', 'values': {'ProductName': 'Evaporative Cooler', 'Value': '3523', 'Weight': '139'}}, {'source': '', 'values': {'ProductName': 'Package Unit', 'Value': '1701', 'Weight': '105'}}]
import gurobipy as gp
from gurobipy import GRB
storage_records = [r for r in LEGACY_RECORDS if 'StorageID' in r['values']]
product_records = [r for r in LEGACY_RECORDS if 'ProductName' in r['values']]
storage_ids = [rec['values']['StorageID'] for rec in storage_records]
capacities = {rec['values']['StorageID']: int(rec['values']['Capacity']) for rec in storage_records}
product_names = [rec['values']['ProductName'] for rec in product_records]
values = {rec['values']['ProductName']: int(rec['values']['Value']) for rec in product_records}
weights = {rec['values']['ProductName']: int(rec['values']['Weight']) for rec in product_records}
if set(capacities.keys()) != set(storage_ids):
    raise ValueError('Mismatch in storage IDs and capacities')
if set(values.keys()) != set(product_names) or set(weights.keys()) != set(product_names):
    raise ValueError('Mismatch in product names, values, or weights')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in storage_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in product_names)) <= capacities[i] for i in storage_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')