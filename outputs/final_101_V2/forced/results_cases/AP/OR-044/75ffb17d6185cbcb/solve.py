import gurobipy as gp
from gurobipy import GRB
sections = {'1': 100, '2': 150, '3': 120, '4': 130, '5': 90, '6': 110, '7': 160, '8': 140}
products = {'1': {'Value': 10, 'Weight': 2}, '2': {'Value': 15, 'Weight': 3}, '3': {'Value': 8, 'Weight': 1}, '4': {'Value': 12, 'Weight': 2}, '5': {'Value': 20, 'Weight': 4}, '6': {'Value': 25, 'Weight': 5}, '7': {'Value': 5, 'Weight': 1}, '8': {'Value': 30, 'Weight': 6}, '9': {'Value': 18, 'Weight': 3}, '10': {'Value': 22, 'Weight': 4}}
section_ids = list(sections.keys())
product_ids = list(products.keys())
for i in section_ids:
    if i not in sections:
        raise ValueError(f'Missing capacity for section {i}')
for j in product_ids:
    if j not in products or 'Value' not in products[j] or 'Weight' not in products[j]:
        raise ValueError(f'Missing value/weight for product {j}')
m = gp.Model('Supermarket_Stocking')
x = m.addVars(section_ids, product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((products[j]['Value'] * x[i, j] for i in section_ids for j in product_ids)), GRB.MAXIMIZE)
for i in section_ids:
    m.addConstr(gp.quicksum((products[j]['Weight'] * x[i, j] for j in product_ids)) <= sections[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')