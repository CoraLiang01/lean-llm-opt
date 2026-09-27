LEGACY_OBSERVATION = '{"values": {"SectionID": "1", "Capacity": "100"}}\n{"values": {"SectionID": "2", "Capacity": "150"}}\n{"values": {"SectionID": "3", "Capacity": "120"}}\n{"values": {"SectionID": "4", "Capacity": "130"}}\n{"values": {"SectionID": "5", "Capacity": "90"}}\n{"values": {"SectionID": "6", "Capacity": "110"}}\n{"values": {"SectionID": "7", "Capacity": "160"}}\n{"values": {"SectionID": "8", "Capacity": "140"}}\n{"values": {"ProductName": "1", "Value": "10", "Weight": "2"}}\n{"values": {"ProductName": "2", "Value": "15", "Weight": "3"}}\n{"values": {"ProductName": "3", "Value": "8", "Weight": "1"}}\n{"values": {"ProductName": "4", "Value": "12", "Weight": "2"}}\n{"values": {"ProductName": "5", "Value": "20", "Weight": "4"}}\n{"values": {"ProductName": "6", "Value": "25", "Weight": "5"}}\n{"values": {"ProductName": "7", "Value": "5", "Weight": "1"}}\n{"values": {"ProductName": "8", "Value": "30", "Weight": "6"}}\n{"values": {"ProductName": "9", "Value": "18", "Weight": "3"}}\n{"values": {"ProductName": "10", "Value": "22", "Weight": "4"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'SectionID': '1', 'Capacity': '100'}}, {'source': '', 'values': {'SectionID': '2', 'Capacity': '150'}}, {'source': '', 'values': {'SectionID': '3', 'Capacity': '120'}}, {'source': '', 'values': {'SectionID': '4', 'Capacity': '130'}}, {'source': '', 'values': {'SectionID': '5', 'Capacity': '90'}}, {'source': '', 'values': {'SectionID': '6', 'Capacity': '110'}}, {'source': '', 'values': {'SectionID': '7', 'Capacity': '160'}}, {'source': '', 'values': {'SectionID': '8', 'Capacity': '140'}}, {'source': '', 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}}, {'source': '', 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}}, {'source': '', 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}}, {'source': '', 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}}, {'source': '', 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}}, {'source': '', 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}}, {'source': '', 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}}, {'source': '', 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}}, {'source': '', 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}]
import gurobipy as gp
from gurobipy import GRB
sections = []
capacities = {}
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'SectionID' in vals and 'Capacity' in vals:
        sid = str(vals['SectionID'])
        sections.append(sid)
        capacities[sid] = int(vals['Capacity'])
    if 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        pid = str(vals['ProductName'])
        products.append(pid)
        values[pid] = int(vals['Value'])
        weights[pid] = int(vals['Weight'])
if len(sections) == 0 or len(products) == 0:
    raise ValueError('Missing section or product data.')
for sid in sections:
    if sid not in capacities:
        raise ValueError(f'Missing capacity for section {sid}')
for pid in products:
    if pid not in values or pid not in weights:
        raise ValueError(f'Missing value or weight for product {pid}')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[pid] * x[sid, pid] for sid in sections for pid in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[pid] * x[sid, pid] for pid in products)) <= capacities[sid] for sid in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')