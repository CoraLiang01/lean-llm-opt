LEGACY_OBSERVATION = '{"values": {"SectionID": "1", "previous_period_capacity": "97", "Capacity": "100"}}\n{"values": {"SectionID": "2", "previous_period_capacity": "163", "Capacity": "150"}}\n{"values": {"SectionID": "3", "previous_period_capacity": "142", "Capacity": "120"}}\n{"values": {"SectionID": "4", "previous_period_capacity": "109", "Capacity": "130"}}\n{"values": {"SectionID": "5", "previous_period_capacity": "79", "Capacity": "90"}}\n{"values": {"SectionID": "6", "previous_period_capacity": "131", "Capacity": "110"}}\n{"values": {"SectionID": "7", "previous_period_capacity": "164", "Capacity": "160"}}\n{"values": {"SectionID": "8", "previous_period_capacity": "153", "Capacity": "140"}}\n{"values": {"previous_period_stock_status": "Overstock", "ProductName": "1", "Value": "10", "previous_period_unit_value": "11", "Weight": "2"}}\n{"values": {"previous_period_stock_status": "Stockout", "ProductName": "2", "Value": "15", "previous_period_unit_value": "12", "Weight": "3"}}\n{"values": {"previous_period_stock_status": "Stockout", "ProductName": "3", "Value": "8", "previous_period_unit_value": "9", "Weight": "1"}}\n{"values": {"previous_period_stock_status": "Balanced", "ProductName": "4", "Value": "12", "previous_period_unit_value": "11", "Weight": "2"}}\n{"values": {"previous_period_stock_status": "Stockout", "ProductName": "5", "Value": "20", "previous_period_unit_value": "16", "Weight": "4"}}\n{"values": {"previous_period_stock_status": "Stockout", "ProductName": "6", "Value": "25", "previous_period_unit_value": "28", "Weight": "5"}}\n{"values": {"previous_period_stock_status": "Balanced", "ProductName": "7", "Value": "5", "previous_period_unit_value": "4", "Weight": "1"}}\n{"values": {"previous_period_stock_status": "Overstock", "ProductName": "8", "Value": "30", "previous_period_unit_value": "27", "Weight": "6"}}\n{"values": {"previous_period_stock_status": "Overstock", "ProductName": "9", "Value": "18", "previous_period_unit_value": "21", "Weight": "3"}}\n{"values": {"previous_period_stock_status": "Balanced", "ProductName": "10", "Value": "22", "previous_period_unit_value": "21", "Weight": "4"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'SectionID': '1', 'previous_period_capacity': '97', 'Capacity': '100'}}, {'source': '', 'values': {'SectionID': '2', 'previous_period_capacity': '163', 'Capacity': '150'}}, {'source': '', 'values': {'SectionID': '3', 'previous_period_capacity': '142', 'Capacity': '120'}}, {'source': '', 'values': {'SectionID': '4', 'previous_period_capacity': '109', 'Capacity': '130'}}, {'source': '', 'values': {'SectionID': '5', 'previous_period_capacity': '79', 'Capacity': '90'}}, {'source': '', 'values': {'SectionID': '6', 'previous_period_capacity': '131', 'Capacity': '110'}}, {'source': '', 'values': {'SectionID': '7', 'previous_period_capacity': '164', 'Capacity': '160'}}, {'source': '', 'values': {'SectionID': '8', 'previous_period_capacity': '153', 'Capacity': '140'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'ProductName': '1', 'Value': '10', 'previous_period_unit_value': '11', 'Weight': '2'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '2', 'Value': '15', 'previous_period_unit_value': '12', 'Weight': '3'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '3', 'Value': '8', 'previous_period_unit_value': '9', 'Weight': '1'}}, {'source': '', 'values': {'previous_period_stock_status': 'Balanced', 'ProductName': '4', 'Value': '12', 'previous_period_unit_value': '11', 'Weight': '2'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '5', 'Value': '20', 'previous_period_unit_value': '16', 'Weight': '4'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '6', 'Value': '25', 'previous_period_unit_value': '28', 'Weight': '5'}}, {'source': '', 'values': {'previous_period_stock_status': 'Balanced', 'ProductName': '7', 'Value': '5', 'previous_period_unit_value': '4', 'Weight': '1'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'ProductName': '8', 'Value': '30', 'previous_period_unit_value': '27', 'Weight': '6'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'ProductName': '9', 'Value': '18', 'previous_period_unit_value': '21', 'Weight': '3'}}, {'source': '', 'values': {'previous_period_stock_status': 'Balanced', 'ProductName': '10', 'Value': '22', 'previous_period_unit_value': '21', 'Weight': '4'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
sections = []
capacity = {}
for rec in records:
    vals = rec['values']
    if 'SectionID' in vals and 'Capacity' in vals:
        sid = str(vals['SectionID'])
        sections.append(sid)
        capacity[sid] = int(vals['Capacity'])
products = []
value = {}
weight = {}
for rec in records:
    vals = rec['values']
    if 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        pid = str(vals['ProductName'])
        products.append(pid)
        value[pid] = int(vals['Value'])
        weight[pid] = int(vals['Weight'])
sections = list(dict.fromkeys(sections))
products = list(dict.fromkeys(products))
for sid in sections:
    if sid not in capacity:
        raise ValueError(f'Missing capacity for section {sid}')
for pid in products:
    if pid not in value or pid not in weight:
        raise ValueError(f'Missing value or weight for product {pid}')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[pid] * x[sid, pid] for sid in sections for pid in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[pid] * x[sid, pid] for pid in products)) <= capacity[sid] for sid in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')