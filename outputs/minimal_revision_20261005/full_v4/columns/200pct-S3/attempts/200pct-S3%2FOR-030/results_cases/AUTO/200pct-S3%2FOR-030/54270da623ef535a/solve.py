import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = list(range(1, 111))
df_ids = set(df['Project ID'].astype(int))
missing_ids = set(project_ids) - df_ids
if missing_ids:
    raise ValueError(f'Missing required Project IDs in CSV: {sorted(missing_ids)}')
capital = df.set_index('Project ID')['Capital (k$)'].astype(int).to_dict()
npv = df.set_index('Project ID')['NPV (k$)'].astype(int).to_dict()
for i in project_ids:
    if i not in capital or i not in npv:
        raise ValueError(f'Missing capital or NPV for Project ID {i}')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in project_ids:
        print(f'x[{i}] {x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')