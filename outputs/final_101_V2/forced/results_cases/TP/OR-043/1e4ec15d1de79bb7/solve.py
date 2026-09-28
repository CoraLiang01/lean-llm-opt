import gurobipy as gp
from gurobipy import GRB
drugs = ['NSAIDs', 'Antirheumatic Drugs', 'Acetic Acid Derivatives', 'Antibiotics', 'Antiviral Drugs', 'Antifungal Agents', 'Antidepressants', 'Antipsychotics', 'Antihistamines', 'Corticosteroids', 'Beta Blockers', 'Calcium Channel Blockers', 'ACE Inhibitors', 'Angiotensin II Receptor Blockers', 'Diuretics', 'Statins', 'Insulin', 'Anticoagulants', 'Antiepileptic Drugs', 'Antiemetics']
benefit = {'NSAIDs': 250, 'Antirheumatic Drugs': 178, 'Acetic Acid Derivatives': 313, 'Antibiotics': 301, 'Antiviral Drugs': 425, 'Antifungal Agents': 260, 'Antidepressants': 848, 'Antipsychotics': 934, 'Antihistamines': 114, 'Corticosteroids': 1357, 'Beta Blockers': 156, 'Calcium Channel Blockers': 1780, 'ACE Inhibitors': 695, 'Angiotensin II Receptor Blockers': 405, 'Diuretics': 320, 'Statins': 320, 'Insulin': 1357, 'Anticoagulants': 1357, 'Antiepileptic Drugs': 405, 'Antiemetics': 998}
weight = {'NSAIDs': 913, 'Antirheumatic Drugs': 754, 'Acetic Acid Derivatives': 428, 'Antibiotics': 711, 'Antiviral Drugs': 350, 'Antifungal Agents': 159, 'Antidepressants': 353, 'Antipsychotics': 291, 'Antihistamines': 302, 'Corticosteroids': 50, 'Beta Blockers': 250, 'Calcium Channel Blockers': 178, 'ACE Inhibitors': 313, 'Angiotensin II Receptor Blockers': 378, 'Diuretics': 94, 'Statins': 97, 'Insulin': 470, 'Anticoagulants': 341, 'Antiepileptic Drugs': 121, 'Antiemetics': 61}
capacity = 520
if set(benefit.keys()) != set(drugs):
    raise ValueError('Benefit data missing for some drugs.')
if set(weight.keys()) != set(drugs):
    raise ValueError('Weight data missing for some drugs.')
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(drugs, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((benefit[i] * x[i] for i in drugs)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in drugs)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')