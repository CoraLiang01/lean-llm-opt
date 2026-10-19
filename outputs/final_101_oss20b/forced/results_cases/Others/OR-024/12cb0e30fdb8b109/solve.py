import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
df = pd.read_csv(data_path, sep=',')
df_s700 = df[df['Product Name'].str.startswith('S700_')]
P = df_s700['Product Name'].tolist()
Revenue = dict(zip(df_s700['Product Name'], df_s700['Revenue']))
Demand = dict(zip(df_s700['Product Name'], df_s700['Demand']))
Inv = dict(zip(df_s700['Product Name'], df_s700['Initial Inventory']))
m = gp.Model('MaximizeRevenue')
x = m.addVars(P, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((Revenue[p] * x[p] for p in P)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= Demand[p] for p in P))
m.addConstrs((x[p] <= Inv[p] for p in P))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Sales Plan ---')
    for p in P:
        sold = int(round(x[p].X))
        print(f'{p}: {sold} units')
else:
    print(f'No optimal solution found. Status: {m.status}')