LEGACY_OBSERVATION = 'capacity.csv\ninventory_audit_staff_count,StorageID,storage_area_cleaning_minutes_last_month,Capacity\n5,1,120,1083\n3,2,300,1840\n2,3,120,770\n5,4,180,1299\n5,5,180,1259\n3,6,180,543\n3,7,120,1831\n5,8,360,855\n5,9,240,619\n5,10,360,637\n3,11,240,935\n4,12,360,626\n6,13,240,1457\n2,14,120,1198\n6,15,360,837\n\nproducts.csv\nenergy_label_review_count,ProductName,supplier_service_tier,warranty_months,Value,Weight\n2,Window Unit,Standard,36,4811,114\n3,Portable Unit,Standard,24,1130,200\n1,Split System,Priority,12,1611,106\n1,Ductless System,Standard,36,3368,256\n2,Central AC,Priority,24,2135,268\n4,Hybrid AC,Priority,24,1046,185\n1,Geothermal AC,Priority,48,4030,299\n6,Smart AC,Premium,12,3761,131\n2,Evaporative Cooler,Premium,12,3523,139\n3,Package Unit,Standard,48,1701,105'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '5', 'StorageID': '1', 'storage_area_cleaning_minutes_last_month': '120', 'Capacity': '1083'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '3', 'StorageID': '2', 'storage_area_cleaning_minutes_last_month': '300', 'Capacity': '1840'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '2', 'StorageID': '3', 'storage_area_cleaning_minutes_last_month': '120', 'Capacity': '770'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '5', 'StorageID': '4', 'storage_area_cleaning_minutes_last_month': '180', 'Capacity': '1299'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '5', 'StorageID': '5', 'storage_area_cleaning_minutes_last_month': '180', 'Capacity': '1259'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '3', 'StorageID': '6', 'storage_area_cleaning_minutes_last_month': '180', 'Capacity': '543'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '3', 'StorageID': '7', 'storage_area_cleaning_minutes_last_month': '120', 'Capacity': '1831'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '5', 'StorageID': '8', 'storage_area_cleaning_minutes_last_month': '360', 'Capacity': '855'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '5', 'StorageID': '9', 'storage_area_cleaning_minutes_last_month': '240', 'Capacity': '619'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '5', 'StorageID': '10', 'storage_area_cleaning_minutes_last_month': '360', 'Capacity': '637'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '3', 'StorageID': '11', 'storage_area_cleaning_minutes_last_month': '240', 'Capacity': '935'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '4', 'StorageID': '12', 'storage_area_cleaning_minutes_last_month': '360', 'Capacity': '626'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '6', 'StorageID': '13', 'storage_area_cleaning_minutes_last_month': '240', 'Capacity': '1457'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '2', 'StorageID': '14', 'storage_area_cleaning_minutes_last_month': '120', 'Capacity': '1198'}}, {'source': 'capacity.csv', 'values': {'inventory_audit_staff_count': '6', 'StorageID': '15', 'storage_area_cleaning_minutes_last_month': '360', 'Capacity': '837'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '2', 'ProductName': 'Window Unit', 'supplier_service_tier': 'Standard', 'warranty_months': '36', 'Value': '4811', 'Weight': '114'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '3', 'ProductName': 'Portable Unit', 'supplier_service_tier': 'Standard', 'warranty_months': '24', 'Value': '1130', 'Weight': '200'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '1', 'ProductName': 'Split System', 'supplier_service_tier': 'Priority', 'warranty_months': '12', 'Value': '1611', 'Weight': '106'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '1', 'ProductName': 'Ductless System', 'supplier_service_tier': 'Standard', 'warranty_months': '36', 'Value': '3368', 'Weight': '256'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '2', 'ProductName': 'Central AC', 'supplier_service_tier': 'Priority', 'warranty_months': '24', 'Value': '2135', 'Weight': '268'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '4', 'ProductName': 'Hybrid AC', 'supplier_service_tier': 'Priority', 'warranty_months': '24', 'Value': '1046', 'Weight': '185'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '1', 'ProductName': 'Geothermal AC', 'supplier_service_tier': 'Priority', 'warranty_months': '48', 'Value': '4030', 'Weight': '299'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '6', 'ProductName': 'Smart AC', 'supplier_service_tier': 'Premium', 'warranty_months': '12', 'Value': '3761', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '2', 'ProductName': 'Evaporative Cooler', 'supplier_service_tier': 'Premium', 'warranty_months': '12', 'Value': '3523', 'Weight': '139'}}, {'source': 'products.csv', 'values': {'energy_label_review_count': '3', 'ProductName': 'Package Unit', 'supplier_service_tier': 'Standard', 'warranty_months': '48', 'Value': '1701', 'Weight': '105'}}]
import gurobipy as gp
from gurobipy import GRB
storage_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
if not storage_records:
    raise ValueError('No capacity.csv records found in LEGACY_RECORDS')
storages = [r['values']['StorageID'] for r in storage_records]
capacities = {}
for r in storage_records:
    sid = r['values']['StorageID']
    cap = r['values']['Capacity']
    if sid in capacities:
        raise ValueError(f'Duplicate StorageID {sid} in capacity.csv')
    capacities[sid] = int(cap)
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
if not product_records:
    raise ValueError('No products.csv records found in LEGACY_RECORDS')
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
    raise ValueError('Mismatch in number of storage areas and capacities')
if len(products) != len(values) or len(products) != len(weights):
    raise ValueError('Mismatch in number of products and value/weight entries')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storages, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in storages for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i] for i in storages), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')