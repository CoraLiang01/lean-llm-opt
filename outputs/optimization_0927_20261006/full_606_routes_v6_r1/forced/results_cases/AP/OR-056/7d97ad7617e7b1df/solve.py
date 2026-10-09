import gurobipy as gp
from gurobipy import GRB
display_areas = [{'DisplayID': '1', 'Capacity': 457}, {'DisplayID': '2', 'Capacity': 604}, {'DisplayID': '3', 'Capacity': 751}, {'DisplayID': '4', 'Capacity': 468}, {'DisplayID': '5', 'Capacity': 343}, {'DisplayID': '6', 'Capacity': 408}, {'DisplayID': '7', 'Capacity': 741}, {'DisplayID': '8', 'Capacity': 914}, {'DisplayID': '9', 'Capacity': 682}, {'DisplayID': '10', 'Capacity': 409}, {'DisplayID': '11', 'Capacity': 342}, {'DisplayID': '12', 'Capacity': 903}, {'DisplayID': '13', 'Capacity': 680}, {'DisplayID': '14', 'Capacity': 886}]
products = [{'ProductName': 'Speedboat', 'Value': 29664, 'Weight': 18}, {'ProductName': 'Fishing Boat', 'Value': 31778, 'Weight': 36}, {'ProductName': 'Catamaran', 'Value': 73501, 'Weight': 25}, {'ProductName': 'Yacht', 'Value': 78255, 'Weight': 16}, {'ProductName': 'Sailboat', 'Value': 93606, 'Weight': 97}, {'ProductName': 'Kayak', 'Value': 46983, 'Weight': 35}, {'ProductName': 'Canoe', 'Value': 95026, 'Weight': 32}, {'ProductName': 'Houseboat', 'Value': 57685, 'Weight': 100}, {'ProductName': 'Pontoon', 'Value': 60323, 'Weight': 43}, {'ProductName': 'Jet Ski', 'Value': 91224, 'Weight': 15}, {'ProductName': 'Rowboat', 'Value': 44003, 'Weight': 95}, {'ProductName': 'Hovercraft', 'Value': 75998, 'Weight': 57}, {'ProductName': 'Cabin Cruiser', 'Value': 84525, 'Weight': 13}, {'ProductName': 'Wakeboard Boat', 'Value': 66207, 'Weight': 44}, {'ProductName': 'Dinghy', 'Value': 65002, 'Weight': 64}, {'ProductName': 'Trawler', 'Value': 33132, 'Weight': 88}, {'ProductName': 'Paddle Boat', 'Value': 69239, 'Weight': 42}, {'ProductName': 'Submarine', 'Value': 66948, 'Weight': 46}, {'ProductName': 'RIB', 'Value': 88240, 'Weight': 24}, {'ProductName': 'Skiff', 'Value': 48858, 'Weight': 93}]
display_ids = [d['DisplayID'] for d in display_areas]
product_names = [p['ProductName'] for p in products]
capacity = {d['DisplayID']: d['Capacity'] for d in display_areas}
value = {p['ProductName']: p['Value'] for p in products}
weight = {p['ProductName']: p['Weight'] for p in products}
m = gp.Model('boat_display_assignment')
x_vars = m.addVars(display_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in product_names)) <= capacity[i] for i in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')