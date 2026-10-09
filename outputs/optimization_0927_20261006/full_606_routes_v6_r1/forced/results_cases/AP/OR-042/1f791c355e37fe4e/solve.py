import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'NSAIDs', 'Value': 585, 'Weight': 50}, {'ProductName': 'Antirheumatic Drugs', 'Value': 557, 'Weight': 329}, {'ProductName': 'Acetic Acid Derivatives', 'Value': 963, 'Weight': 410}, {'ProductName': 'Antibiotics', 'Value': 301, 'Weight': 452}, {'ProductName': 'Antiviral Drugs', 'Value': 425, 'Weight': 350}, {'ProductName': 'Antifungal Agents', 'Value': 260, 'Weight': 159}, {'ProductName': 'Antidepressants', 'Value': 848, 'Weight': 353}, {'ProductName': 'Antipsychotics', 'Value': 461, 'Weight': 291}, {'ProductName': 'Antihistamines', 'Value': 840, 'Weight': 302}, {'ProductName': 'Corticosteroids', 'Value': 999, 'Weight': 50}, {'ProductName': 'Beta Blockers', 'Value': 392, 'Weight': 250}, {'ProductName': 'Calcium Channel Blockers', 'Value': 874, 'Weight': 178}, {'ProductName': 'ACE Inhibitors', 'Value': 695, 'Weight': 313}, {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': 405, 'Weight': 378}, {'ProductName': 'Diuretics', 'Value': 320, 'Weight': 94}, {'ProductName': 'Statins', 'Value': 913, 'Weight': 97}, {'ProductName': 'Insulin', 'Value': 754, 'Weight': 470}, {'ProductName': 'Anticoagulants', 'Value': 428, 'Weight': 341}, {'ProductName': 'Antiepileptic Drugs', 'Value': 711, 'Weight': 121}, {'ProductName': 'Antiemetics', 'Value': 998, 'Weight': 61}]
capacity = 4120
product_names = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
if set(values.keys()) != set(product_names) or set(weights.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers for values or weights.')
m = gp.Model('pharmacy_inventory')
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')