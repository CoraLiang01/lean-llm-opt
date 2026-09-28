import gurobipy as gp
from gurobipy import GRB
display_areas = {'1': 457, '2': 604, '3': 751, '4': 468, '5': 343, '6': 408, '7': 741, '8': 914, '9': 682, '10': 409, '11': 342, '12': 903, '13': 680, '14': 886}
vessel_types = [{'ProductName': 'Speedboat', 'Value': 29664, 'Weight': 18}, {'ProductName': 'Fishing Boat', 'Value': 31778, 'Weight': 36}, {'ProductName': 'Catamaran', 'Value': 73501, 'Weight': 25}, {'ProductName': 'Yacht', 'Value': 78255, 'Weight': 16}, {'ProductName': 'Sailboat', 'Value': 93606, 'Weight': 97}, {'ProductName': 'Kayak', 'Value': 46983, 'Weight': 35}, {'ProductName': 'Canoe', 'Value': 95026, 'Weight': 32}, {'ProductName': 'Houseboat', 'Value': 57685, 'Weight': 100}, {'ProductName': 'Pontoon', 'Value': 60323, 'Weight': 43}, {'ProductName': 'Jet Ski', 'Value': 91224, 'Weight': 15}, {'ProductName': 'Rowboat', 'Value': 44003, 'Weight': 95}, {'ProductName': 'Hovercraft', 'Value': 75998, 'Weight': 57}, {'ProductName': 'Cabin Cruiser', 'Value': 84525, 'Weight': 13}, {'ProductName': 'Wakeboard Boat', 'Value': 66207, 'Weight': 44}, {'ProductName': 'Dinghy', 'Value': 65002, 'Weight': 64}, {'ProductName': 'Trawler', 'Value': 33132, 'Weight': 88}, {'ProductName': 'Paddle Boat', 'Value': 69239, 'Weight': 42}, {'ProductName': 'Submarine', 'Value': 66948, 'Weight': 46}, {'ProductName': 'RIB', 'Value': 88240, 'Weight': 24}, {'ProductName': 'Skiff', 'Value': 48858, 'Weight': 93}]
areas = list(display_areas.keys())
vessels = [v['ProductName'] for v in vessel_types]
vessel_value = {v['ProductName']: v['Value'] for v in vessel_types}
vessel_weight = {v['ProductName']: v['Weight'] for v in vessel_types}
if len(areas) != 14:
    raise ValueError('There must be 14 display areas.')
if len(vessels) != 20:
    raise ValueError('There must be 20 vessel types.')
for i in areas:
    if i not in display_areas:
        raise ValueError(f'Missing capacity for display area {i}.')
for v in vessels:
    if v not in vessel_value or v not in vessel_weight:
        raise ValueError(f'Missing value or weight for vessel type {v}.')
m = gp.Model('Boat_Display_Optimization')
x = m.addVars(areas, vessels, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((vessel_value[j] * x[i, j] for i in areas for j in vessels)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((vessel_weight[j] * x[i, j] for j in vessels)) <= display_areas[i] for i in areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')