import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(path)
workers = [col for col in df.columns if col != 'Owner']
m = gp.Model('WageBalance')
w = m.addVars(workers)
m.addConstr(w['Carpenter'] == 60.0)
for j in workers:
    m.addConstr(w[j] >= 0)
for j in workers:
    m.addConstr(gp.quicksum(df[j]) == 10)
for h in df.index:
    owner = df.loc[h, 'Owner']
    income = gp.quicksum((df.loc[h, j] * w[j] for j in workers if j != owner))
    expenditure = df.loc[h, owner] * w['Carpenter']
    m.addConstr(income == expenditure)
m.setObjective(gp.quicksum((w[j] for j in workers)), sense=gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print('Optimal wages (daily, yuan):')
    for j in workers:
        print(f'{j}: {w[j].X:.4f}')
else:
    print(f'Model status: {m.status}')