import gurobipy as gp
from gurobipy import GRB
display_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
capacities = {1: 457, 2: 604, 3: 751, 4: 468, 5: 343, 6: 408, 7: 741, 8: 914, 9: 682, 10: 409, 11: 342, 12: 903, 13: 680, 14: 886}
boat_types = ['Speedboat', 'Fishing Boat', 'Catamaran', 'Yacht', 'Sailboat', 'Kayak', 'Canoe', 'Houseboat', 'Pontoon', 'Jet Ski', 'Rowboat', 'Hovercraft', 'Cabin Cruiser', 'Wakeboard Boat', 'Dinghy', 'Trawler', 'Paddle Boat', 'Submarine', 'RIB', 'Skiff']
values = {'Speedboat': 29664, 'Fishing Boat': 31778, 'Catamaran': 73501, 'Yacht': 78255, 'Sailboat': 93606, 'Kayak': 46983, 'Canoe': 95026, 'Houseboat': 57685, 'Pontoon': 60323, 'Jet Ski': 91224, 'Rowboat': 44003, 'Hovercraft': 75998, 'Cabin Cruiser': 84525, 'Wakeboard Boat': 66207, 'Dinghy': 65002, 'Trawler': 33132, 'Paddle Boat': 69239, 'Submarine': 66948, 'RIB': 88240, 'Skiff': 48858}
weights = {'Speedboat': 18, 'Fishing Boat': 36, 'Catamaran': 25, 'Yacht': 16, 'Sailboat': 97, 'Kayak': 35, 'Canoe': 32, 'Houseboat': 100, 'Pontoon': 43, 'Jet Ski': 15, 'Rowboat': 95, 'Hovercraft': 57, 'Cabin Cruiser': 13, 'Wakeboard Boat': 44, 'Dinghy': 64, 'Trawler': 88, 'Paddle Boat': 42, 'Submarine': 46, 'RIB': 24, 'Skiff': 93}
for i in display_ids:
    if i not in capacities:
        raise ValueError(f'Missing capacity for display area {i}')
for j in boat_types:
    if j not in values or j not in weights:
        raise ValueError(f'Missing value or weight for boat type {j}')
m = gp.Model('Boat_Display_Optimization')
x_vars = m.addVars(display_ids, boat_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in display_ids for j in boat_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x_vars[i, j] for j in boat_types)) <= capacities[i] for i in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')