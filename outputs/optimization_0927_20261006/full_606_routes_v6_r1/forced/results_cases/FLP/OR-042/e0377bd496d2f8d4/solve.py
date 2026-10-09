import gurobipy as gp
from gurobipy import GRB
drug_types = ['NSAIDs', 'Antirheumatic Drugs', 'Acetic Acid Derivatives', 'Antibiotics', 'Antiviral Drugs', 'Antifungal Agents', 'Antidepressants', 'Antipsychotics', 'Antihistamines', 'Corticosteroids', 'Beta Blockers', 'Calcium Channel Blockers', 'ACE Inhibitors', 'Angiotensin II Receptor Blockers', 'Diuretics', 'Statins', 'Insulin', 'Anticoagulants', 'Antiepileptic Drugs', 'Antiemetics']
benefit = {'NSAIDs': 585, 'Antirheumatic Drugs': 557, 'Acetic Acid Derivatives': 963, 'Antibiotics': 301, 'Antiviral Drugs': 425, 'Antifungal Agents': 260, 'Antidepressants': 848, 'Antipsychotics': 461, 'Antihistamines': 840, 'Corticosteroids': 999, 'Beta Blockers': 392, 'Calcium Channel Blockers': 874, 'ACE Inhibitors': 695, 'Angiotensin II Receptor Blockers': 405, 'Diuretics': 320, 'Statins': 913, 'Insulin': 754, 'Anticoagulants': 428, 'Antiepileptic Drugs': 711, 'Antiemetics': 998}
weight = {'NSAIDs': 50, 'Antirheumatic Drugs': 329, 'Acetic Acid Derivatives': 410, 'Antibiotics': 452, 'Antiviral Drugs': 350, 'Antifungal Agents': 159, 'Antidepressants': 353, 'Antipsychotics': 291, 'Antihistamines': 302, 'Corticosteroids': 50, 'Beta Blockers': 250, 'Calcium Channel Blockers': 178, 'ACE Inhibitors': 313, 'Angiotensin II Receptor Blockers': 378, 'Diuretics': 94, 'Statins': 97, 'Insulin': 470, 'Anticoagulants': 341, 'Antiepileptic Drugs': 121, 'Antiemetics': 61}
capacity = 4120
if set(benefit.keys()) != set(drug_types):
    raise ValueError('Benefit coefficients missing for some drug types.')
if set(weight.keys()) != set(drug_types):
    raise ValueError('Weight coefficients missing for some drug types.')
m = gp.Model('PharmacyDrugOrder')
x_vars = m.addVars(drug_types, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((benefit[i] * x_vars[i] for i in drug_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in drug_types)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')