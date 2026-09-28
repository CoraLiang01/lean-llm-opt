LEGACY_OBSERVATION = '{"values": {"DisplayID": "1", "Capacity": "356"}}\n{"values": {"DisplayID": "2", "Capacity": "478"}}\n{"values": {"DisplayID": "3", "Capacity": "305"}}\n{"values": {"DisplayID": "4", "Capacity": "291"}}\n{"values": {"DisplayID": "5", "Capacity": "168"}}\n{"values": {"DisplayID": "6", "Capacity": "449"}}\n{"values": {"DisplayID": "7", "Capacity": "139"}}\n{"values": {"DisplayID": "8", "Capacity": "383"}}\n{"values": {"DisplayID": "9", "Capacity": "472"}}\n{"values": {"DisplayID": "10", "Capacity": "288"}}\n{"values": {"DisplayID": "11", "Capacity": "320"}}\n{"values": {"DisplayID": "12", "Capacity": "250"}}\n{"values": {"DisplayID": "13", "Capacity": "402"}}\n{"values": {"DisplayID": "14", "Capacity": "293"}}\n{"values": {"ProductName": "Speedboat", "Value": "69978", "Weight": "18"}}\n{"values": {"ProductName": "Fishing Boat", "Value": "54011", "Weight": "42"}}\n{"values": {"ProductName": "Catamaran", "Value": "36352", "Weight": "49"}}\n{"values": {"ProductName": "Yacht", "Value": "51521", "Weight": "42"}}\n{"values": {"ProductName": "Sailboat", "Value": "50415", "Weight": "41"}}\n{"values": {"ProductName": "Kayak", "Value": "76109", "Weight": "48"}}\n{"values": {"ProductName": "Canoe", "Value": "50462", "Weight": "22"}}\n{"values": {"ProductName": "Houseboat", "Value": "28989", "Weight": "29"}}\n{"values": {"ProductName": "Pontoon", "Value": "23318", "Weight": "45"}}\n{"values": {"ProductName": "Jet Ski", "Value": "26142", "Weight": "14"}}\n{"values": {"ProductName": "Rowboat", "Value": "42040", "Weight": "38"}}\n{"values": {"ProductName": "Hovercraft", "Value": "85961", "Weight": "47"}}\n{"values": {"ProductName": "Cabin Cruiser", "Value": "50142", "Weight": "45"}}\n{"values": {"ProductName": "Wakeboard Boat", "Value": "48478", "Weight": "28"}}\n{"values": {"ProductName": "Dinghy", "Value": "60953", "Weight": "24"}}\n{"values": {"ProductName": "Trawler", "Value": "95265", "Weight": "39"}}\n{"values": {"ProductName": "Paddle Boat", "Value": "22839", "Weight": "32"}}\n{"values": {"ProductName": "Submarine", "Value": "90957", "Weight": "36"}}\n{"values": {"ProductName": "RIB", "Value": "84652", "Weight": "14"}}\n{"values": {"ProductName": "Skiff", "Value": "78991", "Weight": "16"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'DisplayID': '1', 'Capacity': '356'}}, {'source': '', 'values': {'DisplayID': '2', 'Capacity': '478'}}, {'source': '', 'values': {'DisplayID': '3', 'Capacity': '305'}}, {'source': '', 'values': {'DisplayID': '4', 'Capacity': '291'}}, {'source': '', 'values': {'DisplayID': '5', 'Capacity': '168'}}, {'source': '', 'values': {'DisplayID': '6', 'Capacity': '449'}}, {'source': '', 'values': {'DisplayID': '7', 'Capacity': '139'}}, {'source': '', 'values': {'DisplayID': '8', 'Capacity': '383'}}, {'source': '', 'values': {'DisplayID': '9', 'Capacity': '472'}}, {'source': '', 'values': {'DisplayID': '10', 'Capacity': '288'}}, {'source': '', 'values': {'DisplayID': '11', 'Capacity': '320'}}, {'source': '', 'values': {'DisplayID': '12', 'Capacity': '250'}}, {'source': '', 'values': {'DisplayID': '13', 'Capacity': '402'}}, {'source': '', 'values': {'DisplayID': '14', 'Capacity': '293'}}, {'source': '', 'values': {'ProductName': 'Speedboat', 'Value': '69978', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': 'Fishing Boat', 'Value': '54011', 'Weight': '42'}}, {'source': '', 'values': {'ProductName': 'Catamaran', 'Value': '36352', 'Weight': '49'}}, {'source': '', 'values': {'ProductName': 'Yacht', 'Value': '51521', 'Weight': '42'}}, {'source': '', 'values': {'ProductName': 'Sailboat', 'Value': '50415', 'Weight': '41'}}, {'source': '', 'values': {'ProductName': 'Kayak', 'Value': '76109', 'Weight': '48'}}, {'source': '', 'values': {'ProductName': 'Canoe', 'Value': '50462', 'Weight': '22'}}, {'source': '', 'values': {'ProductName': 'Houseboat', 'Value': '28989', 'Weight': '29'}}, {'source': '', 'values': {'ProductName': 'Pontoon', 'Value': '23318', 'Weight': '45'}}, {'source': '', 'values': {'ProductName': 'Jet Ski', 'Value': '26142', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': 'Rowboat', 'Value': '42040', 'Weight': '38'}}, {'source': '', 'values': {'ProductName': 'Hovercraft', 'Value': '85961', 'Weight': '47'}}, {'source': '', 'values': {'ProductName': 'Cabin Cruiser', 'Value': '50142', 'Weight': '45'}}, {'source': '', 'values': {'ProductName': 'Wakeboard Boat', 'Value': '48478', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': 'Dinghy', 'Value': '60953', 'Weight': '24'}}, {'source': '', 'values': {'ProductName': 'Trawler', 'Value': '95265', 'Weight': '39'}}, {'source': '', 'values': {'ProductName': 'Paddle Boat', 'Value': '22839', 'Weight': '32'}}, {'source': '', 'values': {'ProductName': 'Submarine', 'Value': '90957', 'Weight': '36'}}, {'source': '', 'values': {'ProductName': 'RIB', 'Value': '84652', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': 'Skiff', 'Value': '78991', 'Weight': '16'}}]
import gurobipy as gp
from gurobipy import GRB
display_areas = []
capacities = {}
boat_types = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'DisplayID' in v and 'Capacity' in v:
        display_areas.append(v['DisplayID'])
        capacities[v['DisplayID']] = int(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        boat_types.append(v['ProductName'])
        values[v['ProductName']] = int(v['Value'])
        weights[v['ProductName']] = int(v['Weight'])
if len(display_areas) == 0 or len(boat_types) == 0:
    raise ValueError('Missing display areas or boat types in LEGACY_RECORDS')
for d in display_areas:
    if d not in capacities:
        raise ValueError(f'Missing capacity for display area {d}')
for b in boat_types:
    if b not in values or b not in weights:
        raise ValueError(f'Missing value or weight for boat type {b}')
m = gp.Model('Boat_Display_Allocation')
x = m.addVars(display_areas, boat_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[b] * x[d, b] for d in display_areas for b in boat_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[b] * x[d, b] for b in boat_types)) <= capacities[d] for d in display_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')