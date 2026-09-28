import gurobipy as gp
from gurobipy import GRB
display_areas = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
capacity = {'1': 457, '2': 604, '3': 751, '4': 468, '5': 343, '6': 408, '7': 741, '8': 914, '9': 682, '10': 409, '11': 342, '12': 903, '13': 680, '14': 886}
vessel_types = ['Speedboat', 'Fishing Boat', 'Catamaran', 'Yacht', 'Sailboat', 'Kayak', 'Canoe', 'Houseboat', 'Pontoon', 'Jet Ski', 'Rowboat', 'Hovercraft', 'Cabin Cruiser', 'Wakeboard Boat', 'Dinghy', 'Trawler', 'Paddle Boat', 'Submarine', 'RIB', 'Skiff']
value = {'Speedboat': 29664, 'Fishing Boat': 31778, 'Catamaran': 73501, 'Yacht': 78255, 'Sailboat': 93606, 'Kayak': 46983, 'Canoe': 95026, 'Houseboat': 57685, 'Pontoon': 60323, 'Jet Ski': 91224, 'Rowboat': 44003, 'Hovercraft': 75998, 'Cabin Cruiser': 84525, 'Wakeboard Boat': 66207, 'Dinghy': 65002, 'Trawler': 33132, 'Paddle Boat': 69239, 'Submarine': 66948, 'RIB': 88240, 'Skiff': 48858}
weight = {'Speedboat': 18, 'Fishing Boat': 36, 'Catamaran': 25, 'Yacht': 16, 'Sailboat': 97, 'Kayak': 35, 'Canoe': 32, 'Houseboat': 100, 'Pontoon': 43, 'Jet Ski': 15, 'Rowboat': 95, 'Hovercraft': 57, 'Cabin Cruiser': 13, 'Wakeboard Boat': 44, 'Dinghy': 64, 'Trawler': 88, 'Paddle Boat': 42, 'Submarine': 46, 'RIB': 24, 'Skiff': 93}
if set(capacity.keys()) != set(display_areas):
    raise ValueError('Capacity data missing for some display areas.')
if set(value.keys()) != set(vessel_types) or set(weight.keys()) != set(vessel_types):
    raise ValueError('Value or weight data missing for some vessel types.')
m = gp.Model('Boat_Display_Assignment')
x = m.addVars(display_areas, vessel_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in display_areas for j in vessel_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in vessel_types)) <= capacity[i] for i in display_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')