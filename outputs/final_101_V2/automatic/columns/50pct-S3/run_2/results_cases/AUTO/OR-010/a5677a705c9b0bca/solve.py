LEGACY_OBSERVATION = 'capacity.csv\nSectionID,previous_period_capacity,Capacity\n1,97,100\n2,163,150\n3,142,120\n4,109,130\n5,79,90\n6,131,110\n7,164,160\n8,153,140\n\nproducts.csv\nprevious_period_stock_status,ProductName,Value,previous_period_unit_value,Weight\nOverstock,1,10,11,2\nStockout,2,15,12,3\nStockout,3,8,9,1\nBalanced,4,12,11,2\nStockout,5,20,16,4\nStockout,6,25,28,5\nBalanced,7,5,4,1\nOverstock,8,30,27,6\nOverstock,9,18,21,3\nBalanced,10,22,21,4'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'SectionID': '1', 'previous_period_capacity': '97', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'SectionID': '2', 'previous_period_capacity': '163', 'Capacity': '150'}}, {'source': 'capacity.csv', 'values': {'SectionID': '3', 'previous_period_capacity': '142', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'SectionID': '4', 'previous_period_capacity': '109', 'Capacity': '130'}}, {'source': 'capacity.csv', 'values': {'SectionID': '5', 'previous_period_capacity': '79', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'SectionID': '6', 'previous_period_capacity': '131', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'SectionID': '7', 'previous_period_capacity': '164', 'Capacity': '160'}}, {'source': 'capacity.csv', 'values': {'SectionID': '8', 'previous_period_capacity': '153', 'Capacity': '140'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'ProductName': '1', 'Value': '10', 'previous_period_unit_value': '11', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '2', 'Value': '15', 'previous_period_unit_value': '12', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '3', 'Value': '8', 'previous_period_unit_value': '9', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'ProductName': '4', 'Value': '12', 'previous_period_unit_value': '11', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '5', 'Value': '20', 'previous_period_unit_value': '16', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'ProductName': '6', 'Value': '25', 'previous_period_unit_value': '28', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'ProductName': '7', 'Value': '5', 'previous_period_unit_value': '4', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'ProductName': '8', 'Value': '30', 'previous_period_unit_value': '27', 'Weight': '6'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'ProductName': '9', 'Value': '18', 'previous_period_unit_value': '21', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'ProductName': '10', 'Value': '22', 'previous_period_unit_value': '21', 'Weight': '4'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
sections = []
capacities = {}
for rec in capacity_records:
    sec = rec['values']['SectionID']
    sections.append(sec)
    capacities[sec] = int(rec['values']['Capacity'])
products = []
values = {}
weights = {}
for rec in product_records:
    prod = rec['values']['ProductName']
    products.append(prod)
    values[prod] = int(rec['values']['Value'])
    weights[prod] = int(rec['values']['Weight'])
if set(sections) != set(capacities.keys()):
    raise ValueError('SectionID mismatch between records and capacities.')
if set(products) != set(values.keys()) or set(products) != set(weights.keys()):
    raise ValueError('ProductName mismatch between records and values/weights.')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i] for i in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')