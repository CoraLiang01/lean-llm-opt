import gurobipy as gp
from gurobipy import GRB
sections = [{'SectionID': '1', 'Capacity': 100}, {'SectionID': '2', 'Capacity': 150}, {'SectionID': '3', 'Capacity': 120}, {'SectionID': '4', 'Capacity': 130}, {'SectionID': '5', 'Capacity': 90}, {'SectionID': '6', 'Capacity': 110}, {'SectionID': '7', 'Capacity': 160}, {'SectionID': '8', 'Capacity': 140}]
products = [{'ProductName': '1', 'Value': 10, 'Weight': 2}, {'ProductName': '2', 'Value': 15, 'Weight': 3}, {'ProductName': '3', 'Value': 8, 'Weight': 1}, {'ProductName': '4', 'Value': 12, 'Weight': 2}, {'ProductName': '5', 'Value': 20, 'Weight': 4}, {'ProductName': '6', 'Value': 25, 'Weight': 5}, {'ProductName': '7', 'Value': 5, 'Weight': 1}, {'ProductName': '8', 'Value': 30, 'Weight': 6}, {'ProductName': '9', 'Value': 18, 'Weight': 3}, {'ProductName': '10', 'Value': 22, 'Weight': 4}]
section_ids = [s['SectionID'] for s in sections]
product_names = [p['ProductName'] for p in products]
section_capacities = {s['SectionID']: s['Capacity'] for s in sections}
product_values = {p['ProductName']: p['Value'] for p in products}
product_weights = {p['ProductName']: p['Weight'] for p in products}
if set(section_ids) != set((str(i) for i in range(1, 9))):
    raise ValueError('Section IDs do not match expected range 1-8')
if set(product_names) != set((str(j) for j in range(1, 11))):
    raise ValueError('Product names do not match expected range 1-10')
for sid in section_ids:
    if sid not in section_capacities:
        raise ValueError(f'Missing capacity for section {sid}')
for pname in product_names:
    if pname not in product_values or pname not in product_weights:
        raise ValueError(f'Missing value or weight for product {pname}')
m = gp.Model('Supermarket_Stocking')
x_vars = m.addVars(section_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[pname] * x_vars[sid, pname] for sid in section_ids for pname in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weights[pname] * x_vars[sid, pname] for pname in product_names)) <= section_capacities[sid] for sid in section_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')