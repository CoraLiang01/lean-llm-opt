LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,StorageID,Capacity\n980,1,1083\n1920,2,1840\n870,3,770\n1130,4,1299\n1415,5,1259\n593,6,543\n1930,7,1831\n860,8,855\n710,9,619\n560,10,637\n891,11,935\n683,12,626\n1699,13,1457\n1294,14,1198\n764,15,837\n\nproducts.csv\nProductName,previous_period_stock_status,previous_period_unit_value,Value,Weight\nWindow Unit,Overstock,5493,4811,114\nPortable Unit,Balanced,1152,1130,200\nSplit System,Overstock,1471,1611,106\nDuctless System,Stockout,3565,3368,256\nCentral AC,Stockout,2027,2135,268\nHybrid AC,Stockout,1087,1046,185\nGeothermal AC,Overstock,3746,4030,299\nSmart AC,Balanced,3342,3761,131\nEvaporative Cooler,Overstock,3373,3523,139\nPackage Unit,Overstock,1816,1701,105'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '980', 'StorageID': '1', 'Capacity': '1083'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1920', 'StorageID': '2', 'Capacity': '1840'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '870', 'StorageID': '3', 'Capacity': '770'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1130', 'StorageID': '4', 'Capacity': '1299'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1415', 'StorageID': '5', 'Capacity': '1259'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '593', 'StorageID': '6', 'Capacity': '543'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1930', 'StorageID': '7', 'Capacity': '1831'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '860', 'StorageID': '8', 'Capacity': '855'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '710', 'StorageID': '9', 'Capacity': '619'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '560', 'StorageID': '10', 'Capacity': '637'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '891', 'StorageID': '11', 'Capacity': '935'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '683', 'StorageID': '12', 'Capacity': '626'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1699', 'StorageID': '13', 'Capacity': '1457'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1294', 'StorageID': '14', 'Capacity': '1198'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '764', 'StorageID': '15', 'Capacity': '837'}}, {'source': 'products.csv', 'values': {'ProductName': 'Window Unit', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '5493', 'Value': '4811', 'Weight': '114'}}, {'source': 'products.csv', 'values': {'ProductName': 'Portable Unit', 'previous_period_stock_status': 'Balanced', 'previous_period_unit_value': '1152', 'Value': '1130', 'Weight': '200'}}, {'source': 'products.csv', 'values': {'ProductName': 'Split System', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '1471', 'Value': '1611', 'Weight': '106'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ductless System', 'previous_period_stock_status': 'Stockout', 'previous_period_unit_value': '3565', 'Value': '3368', 'Weight': '256'}}, {'source': 'products.csv', 'values': {'ProductName': 'Central AC', 'previous_period_stock_status': 'Stockout', 'previous_period_unit_value': '2027', 'Value': '2135', 'Weight': '268'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid AC', 'previous_period_stock_status': 'Stockout', 'previous_period_unit_value': '1087', 'Value': '1046', 'Weight': '185'}}, {'source': 'products.csv', 'values': {'ProductName': 'Geothermal AC', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '3746', 'Value': '4030', 'Weight': '299'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smart AC', 'previous_period_stock_status': 'Balanced', 'previous_period_unit_value': '3342', 'Value': '3761', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'ProductName': 'Evaporative Cooler', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '3373', 'Value': '3523', 'Weight': '139'}}, {'source': 'products.csv', 'values': {'ProductName': 'Package Unit', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '1816', 'Value': '1701', 'Weight': '105'}}]
import gurobipy as gp
from gurobipy import GRB
storage_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
storages = [r['values']['StorageID'] for r in storage_records]
capacities = {}
for r in storage_records:
    sid = r['values']['StorageID']
    cap = r['values']['Capacity']
    if sid in capacities:
        raise ValueError(f'Duplicate StorageID {sid} in capacity.csv')
    capacities[sid] = int(cap)
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
products = [r['values']['ProductName'] for r in product_records]
values = {}
weights = {}
for r in product_records:
    pname = r['values']['ProductName']
    val = r['values']['Value']
    wt = r['values']['Weight']
    if pname in values or pname in weights:
        raise ValueError(f'Duplicate ProductName {pname} in products.csv')
    values[pname] = int(val)
    weights[pname] = int(wt)
if len(storages) != len(capacities):
    raise ValueError('Mismatch in storage area count and capacities')
if len(products) != len(values) or len(products) != len(weights):
    raise ValueError('Mismatch in product count and value/weight data')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storages, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[s, p] for s in storages for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[p] * x[s, p] for p in products)) <= capacities[s] for s in storages), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')