LEGACY_OBSERVATION = '{"values": {"Capacity": "180"}}\n\n{"values": {"ProductName": "Baguette", "Value": "888", "Weight": "4"}}\n\n{"values": {"ProductName": "Croissant", "Value": "134", "Weight": "2"}}\n\n{"values": {"ProductName": "Sourdough", "Value": "129", "Weight": "4"}}\n\n{"values": {"ProductName": "Rye Bread", "Value": "370", "Weight": "3"}}\n\n{"values": {"ProductName": "Brioche", "Value": "921", "Weight": "2"}}\n\n{"values": {"ProductName": "Focaccia", "Value": "765", "Weight": "1"}}\n\n{"values": {"ProductName": "Ciabatta", "Value": "154", "Weight": "2"}}\n\n{"values": {"ProductName": "Pita", "Value": "837", "Weight": "1"}}\n\n{"values": {"ProductName": "Bagel", "Value": "584", "Weight": "3"}}\n\n{"values": {"ProductName": "English Muffin", "Value": "365", "Weight": "3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '180'}}, {'source': '', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
products = []
profit = {}
weight = {}
for rec in records:
    vals = rec['values']
    if 'Capacity' in vals:
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals:
        pname = vals['ProductName']
        products.append(pname)
        profit[pname] = int(vals['Value'])
        weight[pname] = int(vals['Weight'])
if capacity is None:
    raise ValueError('Missing total capacity.')
if set(profit.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Missing profit or weight data for some products.')
m = gp.Model('BakeryOrder')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')