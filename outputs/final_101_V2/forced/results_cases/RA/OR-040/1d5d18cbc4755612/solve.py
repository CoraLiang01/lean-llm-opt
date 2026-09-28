LEGACY_OBSERVATION = '{"values": {"Capacity": "4466"}}\n\n{"values": {"ProductName": "Queens", "Value": "443", "Weight": "104"}}\n\n{"values": {"ProductName": "Brooklyn", "Value": "522", "Weight": "368"}}\n\n{"values": {"ProductName": "Manhattan", "Value": "300", "Weight": "483"}}\n\n{"values": {"ProductName": "Bronx", "Value": "767", "Weight": "165"}}\n\n{"values": {"ProductName": "Staten Island", "Value": "300", "Weight": "105"}}\n\n{"values": {"ProductName": "Harlem", "Value": "309", "Weight": "123"}}\n\n{"values": {"ProductName": "Upper East Side", "Value": "598", "Weight": "131"}}\n\n{"values": {"ProductName": "Lower Manhattan", "Value": "460", "Weight": "341"}}\n\n{"values": {"ProductName": "Midtown", "Value": "318", "Weight": "258"}}\n\n{"values": {"ProductName": "Long Island City", "Value": "126", "Weight": "469"}}\n\n{"values": {"ProductName": "Williamsburg", "Value": "593", "Weight": "387"}}\n\n{"values": {"ProductName": "Bushwick", "Value": "871", "Weight": "425"}}\n\n{"values": {"ProductName": "Flatbush", "Value": "858", "Weight": "482"}}\n\n{"values": {"ProductName": "Greenpoint", "Value": "321", "Weight": "495"}}\n\n{"values": {"ProductName": "Park Slope", "Value": "275", "Weight": "305"}}\n\n{"values": {"ProductName": "Astoria", "Value": "700", "Weight": "377"}}\n\n{"values": {"ProductName": "Jackson Heights", "Value": "685", "Weight": "318"}}\n\n{"values": {"ProductName": "Flushing", "Value": "940", "Weight": "56"}}\n\n{"values": {"ProductName": "Sunnyside", "Value": "522", "Weight": "213"}}\n\n{"values": {"ProductName": "Ditmars", "Value": "763", "Weight": "472"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '4466'}}, {'source': '', 'values': {'ProductName': 'Queens', 'Value': '443', 'Weight': '104'}}, {'source': '', 'values': {'ProductName': 'Brooklyn', 'Value': '522', 'Weight': '368'}}, {'source': '', 'values': {'ProductName': 'Manhattan', 'Value': '300', 'Weight': '483'}}, {'source': '', 'values': {'ProductName': 'Bronx', 'Value': '767', 'Weight': '165'}}, {'source': '', 'values': {'ProductName': 'Staten Island', 'Value': '300', 'Weight': '105'}}, {'source': '', 'values': {'ProductName': 'Harlem', 'Value': '309', 'Weight': '123'}}, {'source': '', 'values': {'ProductName': 'Upper East Side', 'Value': '598', 'Weight': '131'}}, {'source': '', 'values': {'ProductName': 'Lower Manhattan', 'Value': '460', 'Weight': '341'}}, {'source': '', 'values': {'ProductName': 'Midtown', 'Value': '318', 'Weight': '258'}}, {'source': '', 'values': {'ProductName': 'Long Island City', 'Value': '126', 'Weight': '469'}}, {'source': '', 'values': {'ProductName': 'Williamsburg', 'Value': '593', 'Weight': '387'}}, {'source': '', 'values': {'ProductName': 'Bushwick', 'Value': '871', 'Weight': '425'}}, {'source': '', 'values': {'ProductName': 'Flatbush', 'Value': '858', 'Weight': '482'}}, {'source': '', 'values': {'ProductName': 'Greenpoint', 'Value': '321', 'Weight': '495'}}, {'source': '', 'values': {'ProductName': 'Park Slope', 'Value': '275', 'Weight': '305'}}, {'source': '', 'values': {'ProductName': 'Astoria', 'Value': '700', 'Weight': '377'}}, {'source': '', 'values': {'ProductName': 'Jackson Heights', 'Value': '685', 'Weight': '318'}}, {'source': '', 'values': {'ProductName': 'Flushing', 'Value': '940', 'Weight': '56'}}, {'source': '', 'values': {'ProductName': 'Sunnyside', 'Value': '522', 'Weight': '213'}}, {'source': '', 'values': {'ProductName': 'Ditmars', 'Value': '763', 'Weight': '472'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
v = {}
w = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == '' and 'Capacity' in rec['values']:
        if capacity is not None:
            raise ValueError('Multiple capacities found.')
        capacity = int(rec['values']['Capacity'])
    elif rec['source'] == '' and 'ProductName' in rec['values']:
        pname = rec['values']['ProductName']
        products.append(pname)
        try:
            v[pname] = int(rec['values']['Value'])
            w[pname] = int(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid value/weight for {pname}: {e}')
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS.')
if set(v.keys()) != set(products) or set(w.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers and coefficients.')
m = gp.Model('NYC_Development')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((v[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in products)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')