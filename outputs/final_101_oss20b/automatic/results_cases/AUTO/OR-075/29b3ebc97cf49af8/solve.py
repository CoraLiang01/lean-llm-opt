import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(data_path, sep=',')
project_ids = df['Project ID'].tolist()
capital = dict(zip(project_ids, df['Capital (k$)']))
npv = dict(zip(project_ids, df['NPV (k$)']))
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((npv[p] * x[p] for p in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[p] * x[p] for p in project_ids)) <= 1000)
m.addConstr(x[4] + x[7] <= 1)
m.addConstr(x[6] <= x[1])
m.addConstr(x[10] <= x[5])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_npv = m.objVal
    total_capital = sum((capital[p] * x[p].X for p in project_ids))
    selected_projects = [p for p in project_ids if x[p].X > 0.5]
    print(f'Optimal total NPV: {total_npv:.2f} k$')
    print(f'Total capital used: {total_capital:.2f} k$ (Budget limit: 1000 k$)')
    print('Selected projects (Project ID):', selected_projects)
else:
    print(f'No optimal solution found. Status: {m.status}')