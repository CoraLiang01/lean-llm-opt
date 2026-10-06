__lean_models_v2 = []

def __lean_capture_v2(value):
    __lean_models_v2.append(value)
    return value
LEGACY_OBSERVATION = '{"values": {"Capacity": "1576"}}\n\n{"values": {"ProductName": "Sedan", "Value": "1752", "Weight": "15"}}\n\n{"values": {"ProductName": "SUV", "Value": "1856", "Weight": "87"}}\n\n{"values": {"ProductName": "Truck", "Value": "8372", "Weight": "36"}}\n\n{"values": {"ProductName": "Convertible", "Value": "6168", "Weight": "30"}}\n\n{"values": {"ProductName": "Minivan", "Value": "9681", "Weight": "33"}}\n\n{"values": {"ProductName": "Coupe", "Value": "8062", "Weight": "72"}}\n\n{"values": {"ProductName": "Hatchback", "Value": "3895", "Weight": "75"}}\n\n{"values": {"ProductName": "Station Wagon", "Value": "3254", "Weight": "71"}}\n\n{"values": {"ProductName": "Electric Car", "Value": "1701", "Weight": "51"}}\n\n{"values": {"ProductName": "Hybrid Car", "Value": "6799", "Weight": "21"}}\n\n{"values": {"ProductName": "Luxury Sedan", "Value": "2724", "Weight": "97"}}\n\n{"values": {"ProductName": "Sports Car", "Value": "6304", "Weight": "52"}}\n\n{"values": {"ProductName": "Crossover", "Value": "3255", "Weight": "25"}}\n\n{"values": {"ProductName": "Diesel Truck", "Value": "1923", "Weight": "15"}}\n\n{"values": {"ProductName": "Compact SUV", "Value": "4103", "Weight": "54"}}\n\n{"values": {"ProductName": "Luxury SUV", "Value": "4429", "Weight": "57"}}\n\n{"values": {"ProductName": "Cargo Van", "Value": "2663", "Weight": "18"}}\n\n{"values": {"ProductName": "Pickup Truck", "Value": "1691", "Weight": "69"}}\n\n{"values": {"ProductName": "Roadster", "Value": "5632", "Weight": "26"}}\n\n{"values": {"ProductName": "Muscle Car", "Value": "4793", "Weight": "38"}}\n\n{"values": {"ProductName": "Off-road Vehicle", "Value": "1343", "Weight": "31"}}\n\n{"values": {"ProductName": "Camper Van", "Value": "9124", "Weight": "74"}}\n\n{"values": {"ProductName": "Compact Car", "Value": "3652", "Weight": "82"}}\n\n{"values": {"ProductName": "Motorcycle", "Value": "8842", "Weight": "49"}}\n\n{"values": {"ProductName": "Electric SUV", "Value": "9176", "Weight": "64"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '1576'}}, {'source': '', 'values': {'ProductName': 'Sedan', 'Value': '1752', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'SUV', 'Value': '1856', 'Weight': '87'}}, {'source': '', 'values': {'ProductName': 'Truck', 'Value': '8372', 'Weight': '36'}}, {'source': '', 'values': {'ProductName': 'Convertible', 'Value': '6168', 'Weight': '30'}}, {'source': '', 'values': {'ProductName': 'Minivan', 'Value': '9681', 'Weight': '33'}}, {'source': '', 'values': {'ProductName': 'Coupe', 'Value': '8062', 'Weight': '72'}}, {'source': '', 'values': {'ProductName': 'Hatchback', 'Value': '3895', 'Weight': '75'}}, {'source': '', 'values': {'ProductName': 'Station Wagon', 'Value': '3254', 'Weight': '71'}}, {'source': '', 'values': {'ProductName': 'Electric Car', 'Value': '1701', 'Weight': '51'}}, {'source': '', 'values': {'ProductName': 'Hybrid Car', 'Value': '6799', 'Weight': '21'}}, {'source': '', 'values': {'ProductName': 'Luxury Sedan', 'Value': '2724', 'Weight': '97'}}, {'source': '', 'values': {'ProductName': 'Sports Car', 'Value': '6304', 'Weight': '52'}}, {'source': '', 'values': {'ProductName': 'Crossover', 'Value': '3255', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'Diesel Truck', 'Value': '1923', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'Compact SUV', 'Value': '4103', 'Weight': '54'}}, {'source': '', 'values': {'ProductName': 'Luxury SUV', 'Value': '4429', 'Weight': '57'}}, {'source': '', 'values': {'ProductName': 'Cargo Van', 'Value': '2663', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': 'Pickup Truck', 'Value': '1691', 'Weight': '69'}}, {'source': '', 'values': {'ProductName': 'Roadster', 'Value': '5632', 'Weight': '26'}}, {'source': '', 'values': {'ProductName': 'Muscle Car', 'Value': '4793', 'Weight': '38'}}, {'source': '', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '1343', 'Weight': '31'}}, {'source': '', 'values': {'ProductName': 'Camper Van', 'Value': '9124', 'Weight': '74'}}, {'source': '', 'values': {'ProductName': 'Compact Car', 'Value': '3652', 'Weight': '82'}}, {'source': '', 'values': {'ProductName': 'Motorcycle', 'Value': '8842', 'Weight': '49'}}, {'source': '', 'values': {'ProductName': 'Electric SUV', 'Value': '9176', 'Weight': '64'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
products = []
values = {}
weights = {}
for rec in records:
    vals = rec['values']
    if 'Capacity' in vals and vals['Capacity']:
        if capacity is not None:
            raise ValueError('Multiple capacities found')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and vals['ProductName']:
        pname = vals['ProductName']
        try:
            v = int(vals['Value'])
            w = int(vals['Weight'])
        except Exception:
            raise ValueError(f'Invalid Value or Weight for {pname}')
        products.append(pname)
        values[pname] = v
        weights[pname] = w
if capacity is None:
    raise ValueError('No capacity found')
if set(values.keys()) != set(products) or set(weights.keys()) != set(products):
    raise ValueError('Missing value or weight for some products')
m = __lean_capture_v2(gp.Model('Vehicle_Inventory'))
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')