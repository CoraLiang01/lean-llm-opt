LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,Capacity\n681,586\n\nproducts.csv\nprevious_period_development_status,previous_period_unit_value,ProductName,Value,Weight\nPlanned,458,Queens,469,954\nPlanned,275,Brooklyn,290,650\nPlanned,202,Manhattan,236,961\nCompleted,257,Bronx,235,950\nPlanned,741,Staten Island,745,379\nCompleted,657,Harlem,684,776\nIn progress,374,Upper East Side,444,381\nIn progress,201,Lower Manhattan,172,808\nIn progress,901,Midtown,1000,937\nPlanned,277,Long Island City,336,608\nPlanned,547,Williamsburg,546,912\nIn progress,549,Bushwick,535,391\nCompleted,432,Flatbush,539,465\nCompleted,985,Greenpoint,831,490\nIn progress,137,Park Slope,139,918\nCompleted,449,Astoria,432,787\nCompleted,570,Jackson Heights,627,347\nPlanned,582,Flushing,629,274\nCompleted,259,Sunnyside,292,642\nIn progress,1025,Ditmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '681', 'Capacity': '586'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '458', 'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '275', 'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '202', 'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '257', 'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '741', 'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '657', 'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_unit_value': '374', 'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_unit_value': '201', 'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_unit_value': '901', 'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '277', 'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '547', 'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_unit_value': '549', 'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '432', 'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '985', 'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_unit_value': '137', 'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '449', 'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '570', 'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Planned', 'previous_period_unit_value': '582', 'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'Completed', 'previous_period_unit_value': '259', 'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'previous_period_development_status': 'In progress', 'previous_period_unit_value': '1025', 'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
areas = []
value = {}
weight = {}
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        name = rec['values']['ProductName']
        areas.append(name)
        value[name] = int(rec['values']['Value'])
        weight[name] = int(rec['values']['Weight'])
    elif rec['source'] == 'capacity.csv':
        if 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing overall development capacity (Capacity) in LEGACY_RECORDS.')
for a in areas:
    if a not in value or a not in weight:
        raise ValueError(f'Missing value or weight for area {a}.')
m = gp.Model('NYC_Development')
x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[a] * x[a] for a in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[a] * x[a] for a in areas)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')