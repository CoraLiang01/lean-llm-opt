LEGACY_OBSERVATION = '{"values": {"Capacity": "586"}}\n\n{"values": {"ProductName": "Queens", "Value": "469", "Weight": "954"}}\n\n{"values": {"ProductName": "Brooklyn", "Value": "290", "Weight": "650"}}\n\n{"values": {"ProductName": "Manhattan", "Value": "236", "Weight": "961"}}\n\n{"values": {"ProductName": "Bronx", "Value": "235", "Weight": "950"}}\n\n{"values": {"ProductName": "Staten Island", "Value": "745", "Weight": "379"}}\n\n{"values": {"ProductName": "Harlem", "Value": "684", "Weight": "776"}}\n\n{"values": {"ProductName": "Upper East Side", "Value": "444", "Weight": "381"}}\n\n{"values": {"ProductName": "Lower Manhattan", "Value": "172", "Weight": "808"}}\n\n{"values": {"ProductName": "Midtown", "Value": "1000", "Weight": "937"}}\n\n{"values": {"ProductName": "Long Island City", "Value": "336", "Weight": "608"}}\n\n{"values": {"ProductName": "Williamsburg", "Value": "546", "Weight": "912"}}\n\n{"values": {"ProductName": "Bushwick", "Value": "535", "Weight": "391"}}\n\n{"values": {"ProductName": "Flatbush", "Value": "539", "Weight": "465"}}\n\n{"values": {"ProductName": "Greenpoint", "Value": "831", "Weight": "490"}}\n\n{"values": {"ProductName": "Park Slope", "Value": "139", "Weight": "918"}}\n\n{"values": {"ProductName": "Astoria", "Value": "432", "Weight": "787"}}\n\n{"values": {"ProductName": "Jackson Heights", "Value": "627", "Weight": "347"}}\n\n{"values": {"ProductName": "Flushing", "Value": "629", "Weight": "274"}}\n\n{"values": {"ProductName": "Sunnyside", "Value": "292", "Weight": "642"}}\n\n{"values": {"ProductName": "Ditmars", "Value": "978", "Weight": "130"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '586'}}, {'source': '', 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': '', 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': '', 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': '', 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': '', 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': '', 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': '', 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': '', 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': '', 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': '', 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': '', 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': '', 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': '', 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': '', 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': '', 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': '', 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': '', 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': '', 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': '', 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': '', 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
areas = []
value = {}
weight = {}
for rec in records:
    vals = rec['values']
    if 'Capacity' in vals:
        if capacity is not None:
            raise ValueError('Multiple capacities found')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        area = vals['ProductName']
        if area in value or area in weight:
            raise ValueError(f'Duplicate area: {area}')
        areas.append(area)
        value[area] = int(vals['Value'])
        weight[area] = int(vals['Weight'])
if capacity is None:
    raise ValueError('No capacity found')
if set(value.keys()) != set(areas) or set(weight.keys()) != set(areas):
    raise ValueError('Mismatch in area keys')
m = gp.Model('NYC_Development')
x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
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