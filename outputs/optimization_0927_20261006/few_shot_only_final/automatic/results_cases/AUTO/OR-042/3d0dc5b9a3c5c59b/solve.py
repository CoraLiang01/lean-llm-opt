import gurobipy as gp
from gurobipy import GRB
products = ['NSAIDs', 'Antirheumatic Drugs', 'Acetic Acid Derivatives', 'Antibiotics', 'Antiviral Drugs', 'Antifungal Agents', 'Antidepressants', 'Antipsychotics', 'Antihistamines', 'Corticosteroids', 'Beta Blockers', 'Calcium Channel Blockers', 'ACE Inhibitors', 'Angiotensin II Receptor Blockers', 'Diuretics', 'Statins', 'Insulin', 'Anticoagulants', 'Antiepileptic Drugs', 'Antiemetics']
value = {'NSAIDs': 585, 'Antirheumatic Drugs': 557, 'Acetic Acid Derivatives': 963, 'Antibiotics': 301, 'Antiviral Drugs': 425, 'Antifungal Agents': 260, 'Antidepressants': 848, 'Antipsychotics': 461, 'Antihistamines': 840, 'Corticosteroids': 999, 'Beta Blockers': 392, 'Calcium Channel Blockers': 874, 'ACE Inhibitors': 695, 'Angiotensin II Receptor Blockers': 405, 'Diuretics': 320, 'Statins': 913, 'Insulin': 754, 'Anticoagulants': 428, 'Antiepileptic Drugs': 711, 'Antiemetics': 998}
weight = {'NSAIDs': 50, 'Antirheumatic Drugs': 329, 'Acetic Acid Derivatives': 410, 'Antibiotics': 452, 'Antiviral Drugs': 350, 'Antifungal Agents': 159, 'Antidepressants': 353, 'Antipsychotics': 291, 'Antihistamines': 302, 'Corticosteroids': 50, 'Beta Blockers': 250, 'Calcium Channel Blockers': 178, 'ACE Inhibitors': 313, 'Angiotensin II Receptor Blockers': 378, 'Diuretics': 94, 'Statins': 913, 'Insulin': 754, 'Anticoagulants': 428, 'Antiepileptic Drugs': 711, 'Antiemetics': 998, 'Statins': 97, 'Insulin': 470, 'Anticoagulants': 341, 'Antiepileptic Drugs': 121, 'Antiemetics': 61}
weight = {'NSAIDs': 50, 'Antirheumatic Drugs': 329, 'Acetic Acid Derivatives': 410, 'Antibiotics': 452, 'Antiviral Drugs': 350, 'Antifungal Agents': 159, 'Antidepressants': 353, 'Antipsychotics': 291, 'Antihistamines': 302, 'Corticosteroids': 50, 'Beta Blockers': 250, 'Calcium Channel Blockers': 178, 'ACE Inhibitors': 313, 'Angiotensin II Receptor Blockers': 378, 'Diuretics': 94, 'Statins': 97, 'Insulin': 470, 'Anticoagulants': 341, 'Antiepileptic Drugs': 121, 'Antiemetics': 61}
capacity = 4120
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch between products and value/weight keys.')
m = gp.Model('pharmacy_inventory_optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')