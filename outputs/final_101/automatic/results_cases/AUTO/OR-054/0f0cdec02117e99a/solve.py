LEGACY_OBSERVATION = '{"values": {"ShelfID": "1", "Capacity": "750"}}\n{"values": {"ShelfID": "2", "Capacity": "820"}}\n{"values": {"ShelfID": "3", "Capacity": "570"}}\n{"values": {"ShelfID": "4", "Capacity": "800"}}\n{"values": {"ShelfID": "5", "Capacity": "550"}}\n{"values": {"ShelfID": "6", "Capacity": "900"}}\n{"values": {"ShelfID": "7", "Capacity": "650"}}\n{"values": {"ShelfID": "8", "Capacity": "800"}}\n{"values": {"ShelfID": "9", "Capacity": "850"}}\n{"values": {"ShelfID": "10", "Capacity": "900"}}\n{"values": {"ProductName": "1", "Value": "55", "Weight": "10"}}\n{"values": {"ProductName": "2", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "3", "Value": "65", "Weight": "5"}}\n{"values": {"ProductName": "4", "Value": "60", "Weight": "15"}}\n{"values": {"ProductName": "5", "Value": "80", "Weight": "25"}}\n{"values": {"ProductName": "6", "Value": "90", "Weight": "35"}}\n{"values": {"ProductName": "7", "Value": "40", "Weight": "45"}}\n{"values": {"ProductName": "8", "Value": "100", "Weight": "55"}}\n{"values": {"ProductName": "9", "Value": "55", "Weight": "65"}}\n{"values": {"ProductName": "10", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "11", "Value": "110", "Weight": "18"}}\n{"values": {"ProductName": "12", "Value": "50", "Weight": "28"}}\n{"values": {"ProductName": "13", "Value": "60", "Weight": "8"}}\n{"values": {"ProductName": "14", "Value": "120", "Weight": "28"}}\n{"values": {"ProductName": "15", "Value": "70", "Weight": "25"}}\n{"values": {"ProductName": "16", "Value": "110", "Weight": "40"}}\n{"values": {"ProductName": "17", "Value": "50", "Weight": "55"}}\n{"values": {"ProductName": "18", "Value": "60", "Weight": "70"}}\n{"values": {"ProductName": "19", "Value": "120", "Weight": "85"}}\n{"values": {"ProductName": "20", "Value": "100", "Weight": "100"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ShelfID': '1', 'Capacity': '750'}}, {'source': '', 'values': {'ShelfID': '2', 'Capacity': '820'}}, {'source': '', 'values': {'ShelfID': '3', 'Capacity': '570'}}, {'source': '', 'values': {'ShelfID': '4', 'Capacity': '800'}}, {'source': '', 'values': {'ShelfID': '5', 'Capacity': '550'}}, {'source': '', 'values': {'ShelfID': '6', 'Capacity': '900'}}, {'source': '', 'values': {'ShelfID': '7', 'Capacity': '650'}}, {'source': '', 'values': {'ShelfID': '8', 'Capacity': '800'}}, {'source': '', 'values': {'ShelfID': '9', 'Capacity': '850'}}, {'source': '', 'values': {'ShelfID': '10', 'Capacity': '900'}}, {'source': '', 'values': {'ProductName': '1', 'Value': '55', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': '2', 'Value': '75', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': '3', 'Value': '65', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': '6', 'Value': '90', 'Weight': '35'}}, {'source': '', 'values': {'ProductName': '7', 'Value': '40', 'Weight': '45'}}, {'source': '', 'values': {'ProductName': '8', 'Value': '100', 'Weight': '55'}}, {'source': '', 'values': {'ProductName': '9', 'Value': '55', 'Weight': '65'}}, {'source': '', 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': '11', 'Value': '110', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': '12', 'Value': '50', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': '13', 'Value': '60', 'Weight': '8'}}, {'source': '', 'values': {'ProductName': '14', 'Value': '120', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'source': '', 'values': {'ProductName': '17', 'Value': '50', 'Weight': '55'}}, {'source': '', 'values': {'ProductName': '18', 'Value': '60', 'Weight': '70'}}, {'source': '', 'values': {'ProductName': '19', 'Value': '120', 'Weight': '85'}}, {'source': '', 'values': {'ProductName': '20', 'Value': '100', 'Weight': '100'}}]
import gurobipy as gp
from gurobipy import GRB
shelf_caps = {}
product_vals = {}
product_wts = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'ShelfID' in v and 'Capacity' in v:
        shelf_caps[v['ShelfID']] = int(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        product_vals[v['ProductName']] = int(v['Value'])
        product_wts[v['ProductName']] = int(v['Weight'])
shelves = sorted(shelf_caps.keys(), key=lambda x: int(x))
products = sorted(product_vals.keys(), key=lambda x: int(x))
if set(shelf_caps.keys()) != set(shelves):
    raise ValueError('Shelf capacity keys mismatch.')
if set(product_vals.keys()) != set(products) or set(product_wts.keys()) != set(products):
    raise ValueError('Product value/weight keys mismatch.')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_vals[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_wts[j] * x[i, j] for j in products)) <= shelf_caps[i] for i in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')