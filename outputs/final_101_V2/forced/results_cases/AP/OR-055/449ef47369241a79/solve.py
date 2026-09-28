import gurobipy as gp
from gurobipy import GRB
display_areas = [{'DisplayID': '1', 'Capacity': 356}, {'DisplayID': '2', 'Capacity': 478}, {'DisplayID': '3', 'Capacity': 305}, {'DisplayID': '4', 'Capacity': 291}, {'DisplayID': '5', 'Capacity': 168}, {'DisplayID': '6', 'Capacity': 449}, {'DisplayID': '7', 'Capacity': 139}, {'DisplayID': '8', 'Capacity': 383}, {'DisplayID': '9', 'Capacity': 472}, {'DisplayID': '10', 'Capacity': 288}, {'DisplayID': '11', 'Capacity': 320}, {'DisplayID': '12', 'Capacity': 250}, {'DisplayID': '13', 'Capacity': 402}, {'DisplayID': '14', 'Capacity': 293}]
products = [{'ProductName': 'Speedboat', 'Value': 69978, 'Weight': 18}, {'ProductName': 'Fishing Boat', 'Value': 54011, 'Weight': 42}, {'ProductName': 'Catamaran', 'Value': 36352, 'Weight': 49}, {'ProductName': 'Yacht', 'Value': 51521, 'Weight': 42}, {'ProductName': 'Sailboat', 'Value': 50415, 'Weight': 41}, {'ProductName': 'Kayak', 'Value': 76109, 'Weight': 48}, {'ProductName': 'Canoe', 'Value': 50462, 'Weight': 22}, {'ProductName': 'Houseboat', 'Value': 28989, 'Weight': 29}, {'ProductName': 'Pontoon', 'Value': 23318, 'Weight': 45}, {'ProductName': 'Jet Ski', 'Value': 26142, 'Weight': 14}, {'ProductName': 'Rowboat', 'Value': 42040, 'Weight': 38}, {'ProductName': 'Hovercraft', 'Value': 85961, 'Weight': 47}, {'ProductName': 'Cabin Cruiser', 'Value': 50142, 'Weight': 45}, {'ProductName': 'Wakeboard Boat', 'Value': 48478, 'Weight': 28}, {'ProductName': 'Dinghy', 'Value': 60953, 'Weight': 24}, {'ProductName': 'Trawler', 'Value': 95265, 'Weight': 39}, {'ProductName': 'Paddle Boat', 'Value': 22839, 'Weight': 32}, {'ProductName': 'Submarine', 'Value': 90957, 'Weight': 36}, {'ProductName': 'RIB', 'Value': 84652, 'Weight': 14}, {'ProductName': 'Skiff', 'Value': 78991, 'Weight': 16}]
areas = [d['DisplayID'] for d in display_areas]
boats = [p['ProductName'] for p in products]
C = {d['DisplayID']: d['Capacity'] for d in display_areas}
v = {p['ProductName']: p['Value'] for p in products}
w = {p['ProductName']: p['Weight'] for p in products}
if len(areas) != 14 or len(boats) != 20:
    raise ValueError('Incorrect number of display areas or products.')
for a in areas:
    if a not in C:
        raise ValueError(f'Missing capacity for area {a}')
for b in boats:
    if b not in v or b not in w:
        raise ValueError(f'Missing value or weight for boat {b}')
m = gp.Model('Boat_Display_Allocation')
x = m.addVars(areas, boats, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[b] * x[a, b] for a in areas for b in boats)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[b] * x[a, b] for b in boats)) <= C[a] for a in areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')