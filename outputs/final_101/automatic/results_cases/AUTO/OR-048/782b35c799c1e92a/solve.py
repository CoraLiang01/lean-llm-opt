LEGACY_OBSERVATION = '{"values": {"StorageID": "1", "Capacity": "1083"}}\n{"values": {"StorageID": "2", "Capacity": "1840"}}\n{"values": {"StorageID": "3", "Capacity": "770"}}\n{"values": {"StorageID": "4", "Capacity": "1299"}}\n{"values": {"StorageID": "5", "Capacity": "1259"}}\n{"values": {"StorageID": "6", "Capacity": "543"}}\n{"values": {"StorageID": "7", "Capacity": "1831"}}\n{"values": {"StorageID": "8", "Capacity": "855"}}\n{"values": {"StorageID": "9", "Capacity": "619"}}\n{"values": {"StorageID": "10", "Capacity": "637"}}\n{"values": {"StorageID": "11", "Capacity": "935"}}\n{"values": {"StorageID": "12", "Capacity": "626"}}\n{"values": {"StorageID": "13", "Capacity": "1457"}}\n{"values": {"StorageID": "14", "Capacity": "1198"}}\n{"values": {"StorageID": "15", "Capacity": "837"}}\n{"values": {"ProductName": "Window Unit", "Value": "4811", "Weight": "114"}}\n{"values": {"ProductName": "Portable Unit", "Value": "1130", "Weight": "200"}}\n{"values": {"ProductName": "Split System", "Value": "1611", "Weight": "106"}}\n{"values": {"ProductName": "Ductless System", "Value": "3368", "Weight": "256"}}\n{"values": {"ProductName": "Central AC", "Value": "2135", "Weight": "268"}}\n{"values": {"ProductName": "Hybrid AC", "Value": "1046", "Weight": "185"}}\n{"values": {"ProductName": "Geothermal AC", "Value": "4030", "Weight": "299"}}\n{"values": {"ProductName": "Smart AC", "Value": "3761", "Weight": "131"}}\n{"values": {"ProductName": "Evaporative Cooler", "Value": "3523", "Weight": "139"}}\n{"values": {"ProductName": "Package Unit", "Value": "1701", "Weight": "105"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'StorageID': '1', 'Capacity': '1083'}}, {'source': '', 'values': {'StorageID': '2', 'Capacity': '1840'}}, {'source': '', 'values': {'StorageID': '3', 'Capacity': '770'}}, {'source': '', 'values': {'StorageID': '4', 'Capacity': '1299'}}, {'source': '', 'values': {'StorageID': '5', 'Capacity': '1259'}}, {'source': '', 'values': {'StorageID': '6', 'Capacity': '543'}}, {'source': '', 'values': {'StorageID': '7', 'Capacity': '1831'}}, {'source': '', 'values': {'StorageID': '8', 'Capacity': '855'}}, {'source': '', 'values': {'StorageID': '9', 'Capacity': '619'}}, {'source': '', 'values': {'StorageID': '10', 'Capacity': '637'}}, {'source': '', 'values': {'StorageID': '11', 'Capacity': '935'}}, {'source': '', 'values': {'StorageID': '12', 'Capacity': '626'}}, {'source': '', 'values': {'StorageID': '13', 'Capacity': '1457'}}, {'source': '', 'values': {'StorageID': '14', 'Capacity': '1198'}}, {'source': '', 'values': {'StorageID': '15', 'Capacity': '837'}}, {'source': '', 'values': {'ProductName': 'Window Unit', 'Value': '4811', 'Weight': '114'}}, {'source': '', 'values': {'ProductName': 'Portable Unit', 'Value': '1130', 'Weight': '200'}}, {'source': '', 'values': {'ProductName': 'Split System', 'Value': '1611', 'Weight': '106'}}, {'source': '', 'values': {'ProductName': 'Ductless System', 'Value': '3368', 'Weight': '256'}}, {'source': '', 'values': {'ProductName': 'Central AC', 'Value': '2135', 'Weight': '268'}}, {'source': '', 'values': {'ProductName': 'Hybrid AC', 'Value': '1046', 'Weight': '185'}}, {'source': '', 'values': {'ProductName': 'Geothermal AC', 'Value': '4030', 'Weight': '299'}}, {'source': '', 'values': {'ProductName': 'Smart AC', 'Value': '3761', 'Weight': '131'}}, {'source': '', 'values': {'ProductName': 'Evaporative Cooler', 'Value': '3523', 'Weight': '139'}}, {'source': '', 'values': {'ProductName': 'Package Unit', 'Value': '1701', 'Weight': '105'}}]
import gurobipy as gp
from gurobipy import GRB
storage_areas = []
capacities = {}
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'StorageID' in vals and 'Capacity' in vals:
        sid = vals['StorageID']
        storage_areas.append(sid)
        capacities[sid] = int(vals['Capacity'])
    elif 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        pname = vals['ProductName']
        products.append(pname)
        values[pname] = int(vals['Value'])
        weights[pname] = int(vals['Weight'])
if len(storage_areas) == 0 or len(products) == 0:
    raise ValueError('Missing storage areas or products in LEGACY_RECORDS')
for sid in storage_areas:
    if sid not in capacities:
        raise ValueError(f'Missing capacity for storage area {sid}')
for pname in products:
    if pname not in values or pname not in weights:
        raise ValueError(f'Missing value or weight for product {pname}')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_areas, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[s, p] for s in storage_areas for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[p] * x[s, p] for p in products)) <= capacities[s] for s in storage_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')