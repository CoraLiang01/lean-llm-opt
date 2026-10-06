LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,Capacity\n681,586\n\nproducts.csv\nprevious_period_development_status,previous_period_resource_requirement,previous_period_unit_value,ProductName,Value,Weight\nPlanned,866,458,Queens,469,954\nPlanned,764,275,Brooklyn,290,650\nPlanned,913,202,Manhattan,236,961\nCompleted,782,257,Bronx,235,950\nPlanned,413,741,Staten Island,745,379\nCompleted,678,657,Harlem,684,776\nIn progress,319,374,Upper East Side,444,381\nIn progress,887,201,Lower Manhattan,172,808\nIn progress,968,901,Midtown,1000,937\nPlanned,541,277,Long Island City,336,608\nPlanned,954,547,Williamsburg,546,912\nIn progress,452,549,Bushwick,535,391\nCompleted,410,432,Flatbush,539,465\nCompleted,491,985,Greenpoint,831,490\nIn progress,840,137,Park Slope,139,918\nCompleted,811,449,Astoria,432,787\nCompleted,392,570,Jackson Heights,627,347\nPlanned,281,582,Flushing,629,274\nCompleted,565,259,Sunnyside,292,642\nIn progress,145,1025,Ditmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '681', 'Capacity': '586'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '866', 'previous_period_unit_value': '458', 'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '764', 'previous_period_unit_value': '275', 'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '913', 'previous_period_unit_value': '202', 'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '782', 'previous_period_unit_value': '257', 'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '413', 'previous_period_unit_value': '741', 'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '678', 'previous_period_unit_value': '657', 'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_resource_requirement': '319', 'previous_period_unit_value': '374', 'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_resource_requirement': '887', 'previous_period_unit_value': '201', 'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_resource_requirement': '968', 'previous_period_unit_value': '901', 'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '541', 'previous_period_unit_value': '277', 'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '954', 'previous_period_unit_value': '547', 'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_resource_requirement': '452', 'previous_period_unit_value': '549', 'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '410', 'previous_period_unit_value': '432', 'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '491', 'previous_period_unit_value': '985', 'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_resource_requirement': '840', 'previous_period_unit_value': '137', 'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '811', 'previous_period_unit_value': '449', 'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '392', 'previous_period_unit_value': '570', 'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_resource_requirement': '281', 'previous_period_unit_value': '582', 'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_resource_requirement': '565', 'previous_period_unit_value': '259', 'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_resource_requirement': '145', 'previous_period_unit_value': '1025', 'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity_records = [r for r in records if r['source'] == 'capacity.csv']
if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
    raise ValueError('Missing capacity data')
capacity = int(capacity_records[0]['values']['Capacity'])
product_records = [r for r in records if r['source'] == 'products.csv']
areas = []
value = {}
weight = {}
for rec in product_records:
    vals = rec['values']
    area = vals['ProductName']
    if area in areas:
        raise ValueError(f'Duplicate area: {area}')
    areas.append(area)
    if 'Value' not in vals or 'Weight' not in vals:
        raise ValueError(f'Missing Value or Weight for area {area}')
    value[area] = int(vals['Value'])
    weight[area] = int(vals['Weight'])
if len(areas) != 20:
    raise ValueError(f'Expected 20 areas, got {len(areas)}')
for area in areas:
    if area not in value or area not in weight:
        raise ValueError(f'Missing coefficients for area {area}')
m = gp.Model('NYC_Development')
x = m.addVars(areas, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in areas)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')