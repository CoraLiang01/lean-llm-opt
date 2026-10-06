LEGACY_OBSERVATION = 'capacity.csv\npermit_office_staff_count,Capacity\n12,586\n\nproducts.csv\nmarketing_region_group,neighborhood_walkability_score,ProductName,Value,Weight\nCampaign East,83,Queens,469,954\nCampaign West,42,Brooklyn,290,650\nCampaign East,71,Manhattan,236,961\nCampaign Central,65,Bronx,235,950\nCampaign Central,42,Staten Island,745,379\nCampaign West,71,Harlem,684,776\nCampaign West,92,Upper East Side,444,381\nCampaign Central,65,Lower Manhattan,172,808\nCampaign West,71,Midtown,1000,937\nCampaign East,92,Long Island City,336,608\nCampaign Central,56,Williamsburg,546,912\nCampaign Central,83,Bushwick,535,391\nCampaign East,42,Flatbush,539,465\nCampaign West,71,Greenpoint,831,490\nCampaign East,71,Park Slope,139,918\nCampaign West,56,Astoria,432,787\nCampaign West,65,Jackson Heights,627,347\nCampaign Central,83,Flushing,629,274\nCampaign East,83,Sunnyside,292,642\nCampaign East,56,Ditmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'permit_office_staff_count': '12', 'Capacity': '586'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '83', 'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '42', 'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '71', 'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign Central', 'neighborhood_walkability_score': '65', 'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign Central', 'neighborhood_walkability_score': '42', 'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '71', 'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '92', 'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign Central', 'neighborhood_walkability_score': '65', 'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '71', 'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '92', 'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign Central', 'neighborhood_walkability_score': '56', 'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign Central', 'neighborhood_walkability_score': '83', 'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '42', 'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '71', 'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '71', 'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '56', 'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign West', 'neighborhood_walkability_score': '65', 'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign Central', 'neighborhood_walkability_score': '83', 'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '83', 'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'marketing_region_group': 'Campaign East', 'neighborhood_walkability_score': '56', 'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
if not capacity_records:
    raise ValueError('No capacity.csv record found in LEGACY_RECORDS')
C = int(capacity_records[0]['values']['Capacity'])
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
if not product_records:
    raise ValueError('No products.csv records found in LEGACY_RECORDS')
areas = []
value = {}
weight = {}
for rec in product_records:
    name = rec['values']['ProductName']
    areas.append(name)
    try:
        value[name] = int(rec['values']['Value'])
        weight[name] = int(rec['values']['Weight'])
    except KeyError as e:
        raise ValueError(f'Missing Value or Weight for area {name}: {e}')
for i in areas:
    if i not in value or i not in weight:
        raise ValueError(f'Missing value or weight for area {i}')
m = gp.Model('NY_Development')
x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in areas)) <= C, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')