import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
df = pd.read_csv(csv_path, sep=',')
mask = df['Product Name'].str.contains('FAUX', case=False, na=False)
df_faux = df[mask].copy()
if df_faux.empty:
    print('No FAUX products found in the dataset.')
    exit()
P = df_faux['Product Name'].tolist()
revenue = df_faux.set_index('Product Name')['Revenue'].to_dict()
demand = df_faux.set_index('Product Name')['Demand'].to_dict()
inv = df_faux.set_index('Product Name')['Initial Inventory'].to_dict()
m = gp.Model('FAUX_Revenue_Maximization')
x = m.addVars(P, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in P)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in P))
m.addConstrs((x[p] <= inv[p] for p in P))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}\n')
    print('Sales plan for FAUX products:')
    for p in sorted(P):
        units_sold = int(round(x[p].X))
        print(f'  {p}: {units_sold} units sold (revenue per unit: {revenue[p]:.2f})')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')