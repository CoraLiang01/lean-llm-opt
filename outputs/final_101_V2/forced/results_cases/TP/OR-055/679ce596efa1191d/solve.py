import gurobipy as gp
from gurobipy import GRB
display_areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
boat_types = ['Speedboat', 'Fishing Boat', 'Catamaran', 'Yacht', 'Sailboat', 'Kayak', 'Canoe', 'Houseboat', 'Pontoon', 'Jet Ski', 'Rowboat', 'Hovercraft', 'Cabin Cruiser', 'Wakeboard Boat', 'Dinghy', 'Trawler', 'Paddle Boat', 'Submarine', 'RIB', 'Skiff']
capacity = {1: 356, 2: 478, 3: 305, 4: 291, 5: 168, 6: 449, 7: 139, 8: 383, 9: 472, 10: 288, 11: 320, 12: 250, 13: 402, 14: 293}
value = {'Speedboat': 69978, 'Fishing Boat': 54011, 'Catamaran': 36352, 'Yacht': 51521, 'Sailboat': 50415, 'Kayak': 76109, 'Canoe': 50462, 'Houseboat': 28989, 'Pontoon': 23318, 'Jet Ski': 26142, 'Rowboat': 42040, 'Hovercraft': 85961, 'Cabin Cruiser': 50142, 'Wakeboard Boat': 48478, 'Dinghy': 60953, 'Trawler': 95265, 'Paddle Boat': 22839, 'Submarine': 90957, 'RIB': 84652, 'Skiff': 78991}
size = {'Speedboat': 18, 'Fishing Boat': 42, 'Catamaran': 49, 'Yacht': 42, 'Sailboat': 41, 'Kayak': 48, 'Canoe': 22, 'Houseboat': 29, 'Pontoon': 45, 'Jet Ski': 14, 'Rowboat': 38, 'Hovercraft': 47, 'Cabin Cruiser': 45, 'Wakeboard Boat': 28, 'Dinghy': 24, 'Trawler': 39, 'Paddle Boat': 32, 'Submarine': 36, 'RIB': 14, 'Skiff': 16}
if set(capacity.keys()) != set(display_areas):
    raise ValueError('Capacity data missing for some display areas.')
if set(value.keys()) != set(boat_types):
    raise ValueError('Value data missing for some boat types.')
if set(size.keys()) != set(boat_types):
    raise ValueError('Size data missing for some boat types.')
m = gp.Model('Boat_Display_Allocation')
x = m.addVars(display_areas, boat_types, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in display_areas for j in boat_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((size[j] * x[i, j] for j in boat_types)) <= capacity[i] for i in display_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')