import gurobipy as gp
from gurobipy import GRB
products = ['NSAIDs', 'Antirheumatic Drugs', 'Acetic Acid Derivatives', 'Antibiotics', 'Antiviral Drugs', 'Antifungal Agents', 'Antidepressants', 'Antipsychotics', 'Antihistamines', 'Corticosteroids', 'Beta Blockers', 'Calcium Channel Blockers', 'ACE Inhibitors', 'Angiotensin II Receptor Blockers', 'Diuretics', 'Statins', 'Insulin', 'Anticoagulants', 'Antiepileptic Drugs', 'Antiemetics']
benefit = {'NSAIDs': 250, 'Antirheumatic Drugs': 178, 'Acetic Acid Derivatives': 313, 'Antibiotics': 301, 'Antiviral Drugs': 425, 'Antifungal Agents': 260, 'Antidepressants': 848, 'Antipsychotics': 934, 'Antihistamines': 114, 'Corticosteroids': 1357, 'Beta Blockers': 156, 'Calcium Channel Blockers': 1780, 'ACE Inhibitors': 695, 'Angiotensin II Receptor Blockers': 405, 'Diuretics': 320, 'Statins': 320, 'Insulin': 1357, 'Anticoagulants': 1357, 'Antiepileptic Drugs': 405, 'Antiemetics': 998}
resource_consumption = {'NSAIDs': 913, 'Antirheumatic Drugs': 754, 'Acetic Acid Derivatives': 428, 'Antibiotics': 711, 'Antiviral Drugs': 350, 'Antifungal Agents': 159, 'Antidepressants': 353, 'Antipsychotics': 291, 'Antihistamines': 302, 'Corticosteroids': 50, 'Beta Blockers': 250, 'Calcium Channel Blockers': 178, 'ACE Inhibitors': 313, 'Angiotensin II Receptor Blockers': 378, 'Diuretics': 94, 'Statins': 97, 'Insulin': 470, 'Anticoagulants': 341, 'Antiepileptic Drugs': 121, 'Antiemetics': 61}
capacity = 520
if set(benefit.keys()) != set(products):
    raise ValueError('Mismatch between benefit keys and products list')
if set(resource_consumption.keys()) != set(products):
    raise ValueError('Mismatch between resource_consumption keys and products list')
m = gp.Model('pharmacy_inventory_optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_consumption[p] * x_vars[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')