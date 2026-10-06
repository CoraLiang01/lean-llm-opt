LEGACY_OBSERVATION = 'capacity.csv\narchive_revision_number,StorageID,Capacity\n6,1,1083\n5,2,1840\n8,3,770\n2,4,1299\n5,5,1259\n4,6,543\n1,7,1831\n3,8,855\n1,9,619\n6,10,637\n2,11,935\n6,12,626\n1,13,1457\n5,14,1198\n8,15,837\n\nproducts.csv\nProductName,record_keeper_group,archive_revision_number,Value,Weight\nWindow Unit,Team C,5,4811,114\nPortable Unit,Team A,4,1130,200\nSplit System,Team C,5,1611,106\nDuctless System,Team A,1,3368,256\nCentral AC,Team A,1,2135,268\nHybrid AC,Team C,2,1046,185\nGeothermal AC,Team C,9,4030,299\nSmart AC,Team B,3,3761,131\nEvaporative Cooler,Team B,9,3523,139\nPackage Unit,Team C,2,1701,105'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'archive_revision_number': '6', 'StorageID': '1', 'Capacity': '1083'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '5', 'StorageID': '2', 'Capacity': '1840'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '8', 'StorageID': '3', 'Capacity': '770'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '2', 'StorageID': '4', 'Capacity': '1299'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '5', 'StorageID': '5', 'Capacity': '1259'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '4', 'StorageID': '6', 'Capacity': '543'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '1', 'StorageID': '7', 'Capacity': '1831'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '3', 'StorageID': '8', 'Capacity': '855'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '1', 'StorageID': '9', 'Capacity': '619'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '6', 'StorageID': '10', 'Capacity': '637'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '2', 'StorageID': '11', 'Capacity': '935'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '6', 'StorageID': '12', 'Capacity': '626'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '1', 'StorageID': '13', 'Capacity': '1457'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '5', 'StorageID': '14', 'Capacity': '1198'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '8', 'StorageID': '15', 'Capacity': '837'}}, {'source': 'products.csv', 'values': {'ProductName': 'Window Unit', 'record_keeper_group': 'Team C', 'archive_revision_number': '5', 'Value': '4811', 'Weight': '114'}}, {'source': 'products.csv', 'values': {'ProductName': 'Portable Unit', 'record_keeper_group': 'Team A', 'archive_revision_number': '4', 'Value': '1130', 'Weight': '200'}}, {'source': 'products.csv', 'values': {'ProductName': 'Split System', 'record_keeper_group': 'Team C', 'archive_revision_number': '5', 'Value': '1611', 'Weight': '106'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ductless System', 'record_keeper_group': 'Team A', 'archive_revision_number': '1', 'Value': '3368', 'Weight': '256'}}, {'source': 'products.csv', 'values': {'ProductName': 'Central AC', 'record_keeper_group': 'Team A', 'archive_revision_number': '1', 'Value': '2135', 'Weight': '268'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid AC', 'record_keeper_group': 'Team C', 'archive_revision_number': '2', 'Value': '1046', 'Weight': '185'}}, {'source': 'products.csv', 'values': {'ProductName': 'Geothermal AC', 'record_keeper_group': 'Team C', 'archive_revision_number': '9', 'Value': '4030', 'Weight': '299'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smart AC', 'record_keeper_group': 'Team B', 'archive_revision_number': '3', 'Value': '3761', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'ProductName': 'Evaporative Cooler', 'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'Value': '3523', 'Weight': '139'}}, {'source': 'products.csv', 'values': {'ProductName': 'Package Unit', 'record_keeper_group': 'Team C', 'archive_revision_number': '2', 'Value': '1701', 'Weight': '105'}}]
import gurobipy as gp
from gurobipy import GRB
capacities = {}
products = {}
weights = {}
values = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        sid = rec['values']['StorageID']
        cap = int(rec['values']['Capacity'])
        capacities[sid] = cap
    elif rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        values[pname] = val
        weights[pname] = wt
        products[pname] = pname
storage_ids = sorted(capacities.keys(), key=lambda x: int(x))
product_names = list(products.keys())
for sid in storage_ids:
    if sid not in capacities:
        raise ValueError(f'Missing capacity for StorageID {sid}')
for pname in product_names:
    if pname not in values or pname not in weights:
        raise ValueError(f'Missing value or weight for ProductName {pname}')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[pname] * x[sid, pname] for sid in storage_ids for pname in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[pname] * x[sid, pname] for pname in product_names)) <= capacities[sid] for sid in storage_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')