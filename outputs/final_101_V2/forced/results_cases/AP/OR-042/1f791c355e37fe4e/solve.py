import gurobipy as gp
from gurobipy import GRB
products = [{'Product Name': 'NSAIDs', 'Value': 585, 'Weight': 50}, {'Product Name': 'Antirheumatic Drugs', 'Value': 557, 'Weight': 329}, {'Product Name': 'Acetic Acid Derivatives', 'Value': 963, 'Weight': 410}, {'Product Name': 'Antibiotics', 'Value': 301, 'Weight': 452}, {'Product Name': 'Antiviral Drugs', 'Value': 425, 'Weight': 350}, {'Product Name': 'Antifungal Agents', 'Value': 260, 'Weight': 159}, {'Product Name': 'Antidepressants', 'Value': 848, 'Weight': 353}, {'Product Name': 'Antipsychotics', 'Value': 461, 'Weight': 291}, {'Product Name': 'Antihistamines', 'Value': 840, 'Weight': 302}, {'Product Name': 'Corticosteroids', 'Value': 999, 'Weight': 50}, {'Product Name': 'Beta Blockers', 'Value': 392, 'Weight': 250}, {'Product Name': 'Calcium Channel Blockers', 'Value': 874, 'Weight': 178}, {'Product Name': 'ACE Inhibitors', 'Value': 695, 'Weight': 313}, {'Product Name': 'Angiotensin II Receptor Blockers', 'Value': 405, 'Weight': 378}, {'Product Name': 'Diuretics', 'Value': 320, 'Weight': 94}, {'Product Name': 'Statins', 'Value': 913, 'Weight': 97}, {'Product Name': 'Insulin', 'Value': 754, 'Weight': 470}, {'Product Name': 'Anticoagulants', 'Value': 428, 'Weight': 341}, {'Product Name': 'Antiepileptic Drugs', 'Value': 711, 'Weight': 121}, {'Product Name': 'Antiemetics', 'Value': 998, 'Weight': 61}]
capacity = 4120
product_names = [p['Product Name'] for p in products]
value = {p['Product Name']: p['Value'] for p in products}
weight = {p['Product Name']: p['Weight'] for p in products}
if set(value.keys()) != set(product_names) or set(weight.keys()) != set(product_names):
    raise ValueError('Missing value or weight data for some products.')
m = gp.Model('Pharmacy_Inventory_Optimization')
x = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_names)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')