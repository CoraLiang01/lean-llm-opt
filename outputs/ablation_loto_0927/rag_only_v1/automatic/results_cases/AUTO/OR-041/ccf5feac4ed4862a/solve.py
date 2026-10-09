LEGACY_OBSERVATION = '{"values": {"Capacity": "586"}}\n{"values": {"ProductName": "Queens", "Value": "469", "Weight": "954"}}\n{"values": {"ProductName": "Brooklyn", "Value": "290", "Weight": "650"}}\n{"values": {"ProductName": "Manhattan", "Value": "236", "Weight": "961"}}\n{"values": {"ProductName": "Bronx", "Value": "235", "Weight": "950"}}\n{"values": {"ProductName": "Staten Island", "Value": "745", "Weight": "379"}}\n{"values": {"ProductName": "Harlem", "Value": "684", "Weight": "776"}}\n{"values": {"ProductName": "Upper East Side", "Value": "444", "Weight": "381"}}\n{"values": {"ProductName": "Lower Manhattan", "Value": "172", "Weight": "808"}}\n{"values": {"ProductName": "Midtown", "Value": "1000", "Weight": "937"}}\n{"values": {"ProductName": "Long Island City", "Value": "336", "Weight": "608"}}\n{"values": {"ProductName": "Williamsburg", "Value": "546", "Weight": "912"}}\n{"values": {"ProductName": "Bushwick", "Value": "535", "Weight": "391"}}\n{"values": {"ProductName": "Flatbush", "Value": "539", "Weight": "465"}}\n{"values": {"ProductName": "Greenpoint", "Value": "831", "Weight": "490"}}\n{"values": {"ProductName": "Park Slope", "Value": "139", "Weight": "918"}}\n{"values": {"ProductName": "Astoria", "Value": "432", "Weight": "787"}}\n{"values": {"ProductName": "Jackson Heights", "Value": "627", "Weight": "347"}}\n{"values": {"ProductName": "Flushing", "Value": "629", "Weight": "274"}}\n{"values": {"ProductName": "Sunnyside", "Value": "292", "Weight": "642"}}\n{"values": {"ProductName": "Ditmars", "Value": "978", "Weight": "130"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '586'}}, {'source': '', 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': '', 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': '', 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': '', 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': '', 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': '', 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': '', 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': '', 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': '', 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': '', 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': '', 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': '', 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': '', 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': '', 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': '', 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': '', 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': '', 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': '', 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': '', 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': '', 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
from gurobipy import Model, GRB

def solve_development_optimization():
    global LEGACY_RECORDS
    capacity = None
    products = []
    for rec in LEGACY_RECORDS:
        vals = rec['values']
        if 'Capacity' in vals:
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(vals['Capacity'])
        elif 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
            products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) == 0:
        raise ValueError('No products found in LEGACY_RECORDS')
    area_keys = [p['ProductName'] for p in products]
    value = {p['ProductName']: p['Value'] for p in products}
    weight = {p['ProductName']: p['Weight'] for p in products}
    if set(value.keys()) != set(area_keys) or set(weight.keys()) != set(area_keys):
        raise ValueError('Mismatch in product keys and coefficients')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(area_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((value[a] * x[a] for a in area_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[a] * x[a] for a in area_keys)) <= capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for a in area_keys:
            print(x[a].VarName, x[a].X)
    else:
        print('Solver status:', m.Status)
m = solve_development_optimization()