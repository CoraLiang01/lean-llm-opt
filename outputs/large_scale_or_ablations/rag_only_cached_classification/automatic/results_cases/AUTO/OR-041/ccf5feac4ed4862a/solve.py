LEGACY_OBSERVATION = '{"values": {"Capacity": "586"}}\n{"values": {"ProductName": "Queens", "Value": "469", "Weight": "954"}}\n{"values": {"ProductName": "Brooklyn", "Value": "290", "Weight": "650"}}\n{"values": {"ProductName": "Manhattan", "Value": "236", "Weight": "961"}}\n{"values": {"ProductName": "Bronx", "Value": "235", "Weight": "950"}}\n{"values": {"ProductName": "Staten Island", "Value": "745", "Weight": "379"}}\n{"values": {"ProductName": "Harlem", "Value": "684", "Weight": "776"}}\n{"values": {"ProductName": "Upper East Side", "Value": "444", "Weight": "381"}}\n{"values": {"ProductName": "Lower Manhattan", "Value": "172", "Weight": "808"}}\n{"values": {"ProductName": "Midtown", "Value": "1000", "Weight": "937"}}\n{"values": {"ProductName": "Long Island City", "Value": "336", "Weight": "608"}}\n{"values": {"ProductName": "Williamsburg", "Value": "546", "Weight": "912"}}\n{"values": {"ProductName": "Bushwick", "Value": "535", "Weight": "391"}}\n{"values": {"ProductName": "Flatbush", "Value": "539", "Weight": "465"}}\n{"values": {"ProductName": "Greenpoint", "Value": "831", "Weight": "490"}}\n{"values": {"ProductName": "Park Slope", "Value": "139", "Weight": "918"}}\n{"values": {"ProductName": "Astoria", "Value": "432", "Weight": "787"}}\n{"values": {"ProductName": "Jackson Heights", "Value": "627", "Weight": "347"}}\n{"values": {"ProductName": "Flushing", "Value": "629", "Weight": "274"}}\n{"values": {"ProductName": "Sunnyside", "Value": "292", "Weight": "642"}}\n{"values": {"ProductName": "Ditmars", "Value": "978", "Weight": "130"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '586'}}, {'source': '', 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': '', 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': '', 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': '', 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': '', 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': '', 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': '', 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': '', 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': '', 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': '', 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': '', 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': '', 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': '', 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': '', 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': '', 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': '', 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': '', 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': '', 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': '', 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': '', 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
from gurobipy import Model, GRB

def solve_development_optimization():
    global LEGACY_RECORDS
    capacities = []
    products = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == '' and 'Capacity' in rec['values']:
            capacities.append(int(rec['values']['Capacity']))
        elif rec['source'] == '' and 'ProductName' in rec['values']:
            products.append(rec['values'])
    if not capacities:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if not products:
        raise ValueError('No products found in LEGACY_RECORDS')
    if len(capacities) != 1:
        raise ValueError('Expected exactly one capacity value')
    total_capacity = capacities[0]
    area_names = []
    value = {}
    weight = {}
    for prod in products:
        area = prod['ProductName']
        area_names.append(area)
        if 'Value' not in prod or 'Weight' not in prod:
            raise ValueError(f'Missing Value or Weight for area {area}')
        value[area] = int(prod['Value'])
        weight[area] = int(prod['Weight'])
    if len(area_names) != 20:
        raise ValueError(f'Expected 20 areas, got {len(area_names)}')
    if set(value.keys()) != set(area_names) or set(weight.keys()) != set(area_names):
        raise ValueError('Mismatch in area identifiers for value/weight')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(area_names, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((value[area] * x[area] for area in area_names)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[area] * x[area] for area in area_names)) <= total_capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for area in area_names:
            print(x[area].VarName, x[area].X)
    else:
        print('Status', m.Status)
    return m
m = solve_development_optimization()