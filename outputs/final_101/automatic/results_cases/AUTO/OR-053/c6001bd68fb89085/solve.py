LEGACY_OBSERVATION = '{"values": {"ShelfID": "1", "Capacity": "500"}}\n{"values": {"ShelfID": "2", "Capacity": "700"}}\n{"values": {"ShelfID": "3", "Capacity": "600"}}\n{"values": {"ShelfID": "4", "Capacity": "800"}}\n{"values": {"ShelfID": "5", "Capacity": "550"}}\n{"values": {"ShelfID": "6", "Capacity": "900"}}\n{"values": {"ShelfID": "7", "Capacity": "650"}}\n{"values": {"ShelfID": "8", "Capacity": "750"}}\n{"values": {"ShelfID": "9", "Capacity": "820"}}\n{"values": {"ShelfID": "10", "Capacity": "570"}}\n{"values": {"ProductName": "1", "Value": "50", "Weight": "10"}}\n{"values": {"ProductName": "2", "Value": "70", "Weight": "20"}}\n{"values": {"ProductName": "3", "Value": "30", "Weight": "5"}}\n{"values": {"ProductName": "4", "Value": "60", "Weight": "15"}}\n{"values": {"ProductName": "5", "Value": "80", "Weight": "25"}}\n{"values": {"ProductName": "6", "Value": "90", "Weight": "30"}}\n{"values": {"ProductName": "7", "Value": "40", "Weight": "12"}}\n{"values": {"ProductName": "8", "Value": "100", "Weight": "35"}}\n{"values": {"ProductName": "9", "Value": "55", "Weight": "10"}}\n{"values": {"ProductName": "10", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "11", "Value": "65", "Weight": "18"}}\n{"values": {"ProductName": "12", "Value": "95", "Weight": "28"}}\n{"values": {"ProductName": "13", "Value": "45", "Weight": "8"}}\n{"values": {"ProductName": "14", "Value": "85", "Weight": "22"}}\n{"values": {"ProductName": "15", "Value": "70", "Weight": "25"}}\n{"values": {"ProductName": "16", "Value": "110", "Weight": "40"}}\n{"values": {"ProductName": "17", "Value": "50", "Weight": "14"}}\n{"values": {"ProductName": "18", "Value": "60", "Weight": "16"}}\n{"values": {"ProductName": "19", "Value": "120", "Weight": "50"}}\n{"values": {"ProductName": "20", "Value": "100", "Weight": "30"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ShelfID': '1', 'Capacity': '500'}}, {'source': '', 'values': {'ShelfID': '2', 'Capacity': '700'}}, {'source': '', 'values': {'ShelfID': '3', 'Capacity': '600'}}, {'source': '', 'values': {'ShelfID': '4', 'Capacity': '800'}}, {'source': '', 'values': {'ShelfID': '5', 'Capacity': '550'}}, {'source': '', 'values': {'ShelfID': '6', 'Capacity': '900'}}, {'source': '', 'values': {'ShelfID': '7', 'Capacity': '650'}}, {'source': '', 'values': {'ShelfID': '8', 'Capacity': '750'}}, {'source': '', 'values': {'ShelfID': '9', 'Capacity': '820'}}, {'source': '', 'values': {'ShelfID': '10', 'Capacity': '570'}}, {'source': '', 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}}, {'source': '', 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}}, {'source': '', 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}}, {'source': '', 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}}, {'source': '', 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}}, {'source': '', 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'source': '', 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}}, {'source': '', 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}]
import gurobipy as gp
from gurobipy import GRB
shelf_caps = {}
shelf_ids = []
product_vals = {}
product_wts = {}
product_names = []
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'ShelfID' in vals and 'Capacity' in vals:
        shelf_id = str(vals['ShelfID'])
        shelf_ids.append(shelf_id)
        shelf_caps[shelf_id] = int(vals['Capacity'])
    if 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        prod = str(vals['ProductName'])
        product_names.append(prod)
        product_vals[prod] = int(vals['Value'])
        product_wts[prod] = int(vals['Weight'])
shelf_ids.sort(key=lambda x: int(x))
product_names.sort(key=lambda x: int(x))
if len(shelf_ids) == 0 or len(product_names) == 0:
    raise ValueError('Missing shelf or product data.')
for sid in shelf_ids:
    if sid not in shelf_caps:
        raise ValueError(f'Missing capacity for shelf {sid}')
for pname in product_names:
    if pname not in product_vals or pname not in product_wts:
        raise ValueError(f'Missing value/weight for product {pname}')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelf_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_vals[p] * x[s, p] for s in shelf_ids for p in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_wts[p] * x[s, p] for p in product_names)) <= shelf_caps[s] for s in shelf_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')