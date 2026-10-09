import gurobipy as gp
from gurobipy import GRB
display_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
capacities = {1: 356, 2: 478, 3: 305, 4: 291, 5: 168, 6: 449, 7: 139, 8: 383, 9: 472, 10: 288, 11: 320, 12: 250, 13: 402, 14: 293}
product_names = ['Speedboat', 'Fishing Boat', 'Catamaran', 'Yacht', 'Sailboat', 'Kayak', 'Canoe', 'Houseboat', 'Pontoon', 'Jet Ski', 'Rowboat', 'Hovercraft', 'Cabin Cruiser', 'Wakeboard Boat', 'Dinghy', 'Trawler', 'Paddle Boat', 'Submarine', 'RIB', 'Skiff']
values = {'Speedboat': 69978, 'Fishing Boat': 54011, 'Catamaran': 36352, 'Yacht': 51521, 'Sailboat': 50415, 'Kayak': 76109, 'Canoe': 50462, 'Houseboat': 28989, 'Pontoon': 23318, 'Jet Ski': 26142, 'Rowboat': 42040, 'Hovercraft': 85961, 'Cabin Cruiser': 50142, 'Wakeboard Boat': 48478, 'Dinghy': 60953, 'Trawler': 95265, 'Paddle Boat': 22839, 'Submarine': 90957, 'RIB': 84652, 'Skiff': 78991}
weights = {'Speedboat': 18, 'Fishing Boat': 42, 'Catamaran': 49, 'Yacht': 42, 'Sailboat': 41, 'Kayak': 48, 'Canoe': 22, 'Houseboat': 29, 'Pontoon': 45, 'Jet Ski': 14, 'Rowboat': 38, 'Hovercraft': 47, 'Cabin Cruiser': 45, 'Wakeboard Boat': 28, 'Dinghy': 24, 'Trawler': 39, 'Paddle Boat': 32, 'Submarine': 36, 'RIB': 14, 'Skiff': 16}
if set(capacities.keys()) != set(display_ids):
    raise ValueError('Missing or extra display IDs in capacities.')
if set(values.keys()) != set(product_names):
    raise ValueError('Missing or extra product names in values.')
if set(weights.keys()) != set(product_names):
    raise ValueError('Missing or extra product names in weights.')
m = gp.Model('Boat_Display_Allocation')
x_vars = m.addVars(display_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x_vars[i, j] for j in product_names)) <= capacities[i] for i in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')