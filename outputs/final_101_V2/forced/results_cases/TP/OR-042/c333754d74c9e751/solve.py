import gurobipy as gp
from gurobipy import GRB
drug_types = [{'id': 1, 'name': 'NSAIDs', 'value': 585, 'weight': 50}, {'id': 2, 'name': 'Antirheumatic Drugs', 'value': 557, 'weight': 329}, {'id': 3, 'name': 'Acetic Acid Derivatives', 'value': 963, 'weight': 410}, {'id': 4, 'name': 'Antibiotics', 'value': 301, 'weight': 452}, {'id': 5, 'name': 'Antiviral Drugs', 'value': 425, 'weight': 350}, {'id': 6, 'name': 'Antifungal Agents', 'value': 260, 'weight': 159}, {'id': 7, 'name': 'Antidepressants', 'value': 848, 'weight': 353}, {'id': 8, 'name': 'Antipsychotics', 'value': 461, 'weight': 291}, {'id': 9, 'name': 'Antihistamines', 'value': 840, 'weight': 302}, {'id': 10, 'name': 'Corticosteroids', 'value': 999, 'weight': 50}, {'id': 11, 'name': 'Beta Blockers', 'value': 392, 'weight': 250}, {'id': 12, 'name': 'Calcium Channel Blockers', 'value': 874, 'weight': 178}, {'id': 13, 'name': 'ACE Inhibitors', 'value': 695, 'weight': 313}, {'id': 14, 'name': 'Angiotensin II Receptor Blockers', 'value': 405, 'weight': 378}, {'id': 15, 'name': 'Diuretics', 'value': 320, 'weight': 94}, {'id': 16, 'name': 'Statins', 'value': 913, 'weight': 97}, {'id': 17, 'name': 'Insulin', 'value': 754, 'weight': 470}, {'id': 18, 'name': 'Anticoagulants', 'value': 428, 'weight': 341}, {'id': 19, 'name': 'Antiepileptic Drugs', 'value': 711, 'weight': 121}, {'id': 20, 'name': 'Antiemetics', 'value': 998, 'weight': 61}]
capacity = 4120
drug_ids = [d['id'] for d in drug_types]
values = {d['id']: d['value'] for d in drug_types}
weights = {d['id']: d['weight'] for d in drug_types}
if not len(drug_ids) == len(values) == len(weights) == 20:
    raise ValueError('Data coverage error: Check drug_types data.')
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(drug_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in drug_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in drug_ids)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in drug_ids:
        print(f'x[{i}]: {x[i].X}')
else:
    print(f'Solver status: {m.Status}')