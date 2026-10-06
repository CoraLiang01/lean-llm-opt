LEGACY_OBSERVATION = 'capacity.csv\n\nprevious_period_capacity,StorageID,capacity_two_periods_ago,Capacity\n980,1,1185,1083\n1920,2,2027,1840\n870,3,796,770\n1130,4,1205,1299\n1415,5,1087,1259\n593,6,651,543\n1930,7,1975,1831\n860,8,873,855\n710,9,626,619\n560,10,654,637\n891,11,884,935\n683,12,515,626\n1699,13,1729,1457\n1294,14,994,1198\n764,15,919,837\n\nproducts.csv\n\nprevious_period_resource_requirement,ProductName,previous_period_stock_status,previous_period_unit_value,Value,Weight\n111,Window Unit,Overstock,5493,4811,114\n231,Portable Unit,Balanced,1152,1130,200\n118,Split System,Overstock,1471,1611,106\n303,Ductless System,Stockout,3565,3368,256\n292,Central AC,Stockout,2027,2135,268\n181,Hybrid AC,Stockout,1087,1046,185\n318,Geothermal AC,Overstock,3746,4030,299\n112,Smart AC,Balanced,3342,3761,131\n163,Evaporative Cooler,Overstock,3373,3523,139\n113,Package Unit,Overstock,1816,1701,105'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '980', 'StorageID': '1', 'capacity_two_periods_ago': '1185', 'Capacity': '1083'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1920', 'StorageID': '2', 'capacity_two_periods_ago': '2027', 'Capacity': '1840'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '870', 'StorageID': '3', 'capacity_two_periods_ago': '796', 'Capacity': '770'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1130', 'StorageID': '4', 'capacity_two_periods_ago': '1205', 'Capacity': '1299'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1415', 'StorageID': '5', 'capacity_two_periods_ago': '1087', 'Capacity': '1259'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '593', 'StorageID': '6', 'capacity_two_periods_ago': '651', 'Capacity': '543'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1930', 'StorageID': '7', 'capacity_two_periods_ago': '1975', 'Capacity': '1831'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '860', 'StorageID': '8', 'capacity_two_periods_ago': '873', 'Capacity': '855'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '710', 'StorageID': '9', 'capacity_two_periods_ago': '626', 'Capacity': '619'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '560', 'StorageID': '10', 'capacity_two_periods_ago': '654', 'Capacity': '637'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '891', 'StorageID': '11', 'capacity_two_periods_ago': '884', 'Capacity': '935'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '683', 'StorageID': '12', 'capacity_two_periods_ago': '515', 'Capacity': '626'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1699', 'StorageID': '13', 'capacity_two_periods_ago': '1729', 'Capacity': '1457'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1294', 'StorageID': '14', 'capacity_two_periods_ago': '994', 'Capacity': '1198'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '764', 'StorageID': '15', 'capacity_two_periods_ago': '919', 'Capacity': '837'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '111', 'ProductName': 'Window Unit', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '5493', 'Value': '4811', 'Weight': '114'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '231', 'ProductName': 'Portable Unit', 'previous_period_stock_status': 'Balanced', 'previous_period_unit_value': '1152', 'Value': '1130', 'Weight': '200'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '118', 'ProductName': 'Split System', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '1471', 'Value': '1611', 'Weight': '106'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '303', 'ProductName': 'Ductless System', 'previous_period_stock_status': 'Stockout', 'previous_period_unit_value': '3565', 'Value': '3368', 'Weight': '256'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '292', 'ProductName': 'Central AC', 'previous_period_stock_status': 'Stockout', 'previous_period_unit_value': '2027', 'Value': '2135', 'Weight': '268'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '181', 'ProductName': 'Hybrid AC', 'previous_period_stock_status': 'Stockout', 'previous_period_unit_value': '1087', 'Value': '1046', 'Weight': '185'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '318', 'ProductName': 'Geothermal AC', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '3746', 'Value': '4030', 'Weight': '299'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '112', 'ProductName': 'Smart AC', 'previous_period_stock_status': 'Balanced', 'previous_period_unit_value': '3342', 'Value': '3761', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '163', 'ProductName': 'Evaporative Cooler', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '3373', 'Value': '3523', 'Weight': '139'}}, {'source': 'products.csv', 'values': {'previous_period_resource_requirement': '113', 'ProductName': 'Package Unit', 'previous_period_stock_status': 'Overstock', 'previous_period_unit_value': '1816', 'Value': '1701', 'Weight': '105'}}]
import gurobipy as gp
from gurobipy import GRB
storage_areas = []
capacities = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        sid = rec['values']['StorageID']
        storage_areas.append(sid)
        capacities[sid] = int(rec['values']['Capacity'])
product_types = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        product_types.append(pname)
        values[pname] = int(rec['values']['Value'])
        weights[pname] = int(rec['values']['Weight'])
if len(storage_areas) == 0 or len(product_types) == 0:
    raise ValueError('Missing storage areas or product types.')
for sid in storage_areas:
    if sid not in capacities:
        raise ValueError(f'Missing capacity for storage area {sid}.')
for pname in product_types:
    if pname not in values or pname not in weights:
        raise ValueError(f'Missing value or weight for product {pname}.')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_areas, product_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in storage_areas for j in product_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in product_types)) <= capacities[i] for i in storage_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')