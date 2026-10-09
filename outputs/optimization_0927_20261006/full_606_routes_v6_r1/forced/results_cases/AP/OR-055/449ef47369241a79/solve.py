import gurobipy as gp
from gurobipy import GRB
display_areas = [{'DisplayID': '1', 'Capacity': 356}, {'DisplayID': '2', 'Capacity': 478}, {'DisplayID': '3', 'Capacity': 305}, {'DisplayID': '4', 'Capacity': 291}, {'DisplayID': '5', 'Capacity': 168}, {'DisplayID': '6', 'Capacity': 449}, {'DisplayID': '7', 'Capacity': 139}, {'DisplayID': '8', 'Capacity': 383}, {'DisplayID': '9', 'Capacity': 472}, {'DisplayID': '10', 'Capacity': 288}, {'DisplayID': '11', 'Capacity': 320}, {'DisplayID': '12', 'Capacity': 250}, {'DisplayID': '13', 'Capacity': 402}, {'DisplayID': '14', 'Capacity': 293}]
products = [{'ProductName': 'Speedboat', 'Value': 69978, 'Weight': 18}, {'ProductName': 'Fishing Boat', 'Value': 54011, 'Weight': 42}, {'ProductName': 'Catamaran', 'Value': 36352, 'Weight': 49}, {'ProductName': 'Yacht', 'Value': 51521, 'Weight': 42}, {'ProductName': 'Sailboat', 'Value': 50415, 'Weight': 41}, {'ProductName': 'Kayak', 'Value': 76109, 'Weight': 48}, {'ProductName': 'Canoe', 'Value': 50462, 'Weight': 22}, {'ProductName': 'Houseboat', 'Value': 28989, 'Weight': 29}, {'ProductName': 'Pontoon', 'Value': 23318, 'Weight': 45}, {'ProductName': 'Jet Ski', 'Value': 26142, 'Weight': 14}, {'ProductName': 'Rowboat', 'Value': 42040, 'Weight': 38}, {'ProductName': 'Hovercraft', 'Value': 85961, 'Weight': 47}, {'ProductName': 'Cabin Cruiser', 'Value': 50142, 'Weight': 45}, {'ProductName': 'Wakeboard Boat', 'Value': 48478, 'Weight': 28}, {'ProductName': 'Dinghy', 'Value': 60953, 'Weight': 24}, {'ProductName': 'Trawler', 'Value': 95265, 'Weight': 39}, {'ProductName': 'Paddle Boat', 'Value': 22839, 'Weight': 32}, {'ProductName': 'Submarine', 'Value': 90957, 'Weight': 36}, {'ProductName': 'RIB', 'Value': 84652, 'Weight': 14}, {'ProductName': 'Skiff', 'Value': 78991, 'Weight': 16}]
display_ids = [area['DisplayID'] for area in display_areas]
product_names = [prod['ProductName'] for prod in products]
capacities = {area['DisplayID']: area['Capacity'] for area in display_areas}
values = {prod['ProductName']: prod['Value'] for prod in products}
weights = {prod['ProductName']: prod['Weight'] for prod in products}
m = gp.Model('Boat_Display_Allocation')
x_vars = m.addVars(display_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x_vars[i, j] for j in product_names)) <= capacities[i] for i in display_ids), name='')
for i in display_ids:
    if i not in capacities:
        raise ValueError(f'Missing capacity for display area {i}')
for j in product_names:
    if j not in values or j not in weights:
        raise ValueError(f'Missing value or weight for product {j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')