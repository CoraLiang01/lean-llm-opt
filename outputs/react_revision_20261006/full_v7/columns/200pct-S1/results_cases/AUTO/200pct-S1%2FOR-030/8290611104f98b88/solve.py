import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(float)
df['NPV (k$)'] = df['NPV (k$)'].astype(float)
project_ids = df['Project ID'].tolist()
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
required_projects = {1: 'Infrastructure Upgrade', 4: 'R&D Initiative Alpha', 5: 'Staff Training Program', 6: 'System Automation', 7: 'Global Expansion Pilot', 10: 'Customer Experience Platform'}
for (pid, pname) in required_projects.items():
    if pid not in project_ids:
        raise ValueError(f"Required project ID {pid} ('{pname}') not found in project.csv")
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in project_ids:
        print(f'y[{i}] {y_vars[i].VarName} {y_vars[i].X}')
else:
    print(f'Solver status: {m.status}')