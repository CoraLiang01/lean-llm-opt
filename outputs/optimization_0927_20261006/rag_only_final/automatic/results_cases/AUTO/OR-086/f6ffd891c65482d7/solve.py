import pandas as pd
import numpy as np
import re
from gurobipy import Model, GRB, quicksum
grades_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
brands_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
grades_df = pd.read_csv(grades_path, dtype=str, keep_default_na=False)
grades_df['Daily Supply (kg)'] = grades_df['Daily Supply (kg)'].astype(float)
grades_df['Cost (CNY/kg)'] = grades_df['Cost (CNY/kg)'].astype(float)
grades_df['Grade'] = grades_df['Grade'].str.strip()
brands_df = pd.read_csv(brands_path, dtype=str, keep_default_na=False)
brands_df['Selling Price (CNY/kg)'] = brands_df['Selling Price (CNY/kg)'].astype(float)
brands_df['Brand'] = brands_df['Brand'].str.strip()
G = list(grades_df['Grade'])
B = list(brands_df['Brand'])
cost_g = {row['Grade']: row['Cost (CNY/kg)'] for (_, row) in grades_df.iterrows()}
supply_g = {row['Grade']: row['Daily Supply (kg)'] for (_, row) in grades_df.iterrows()}
price_b = {row['Brand']: row['Selling Price (CNY/kg)'] for (_, row) in brands_df.iterrows()}
lower_bound = {b: {g: 0.0 for g in G} for b in B}
upper_bound = {b: {g: 1.0 for g in G} for b in B}
less_than_pat = re.compile('([A-Za-z0-9]+)\\s+less than\\s+([0-9]+)%', re.IGNORECASE)
more_than_pat = re.compile('([A-Za-z0-9]+)\\s+more than\\s+([0-9]+)%', re.IGNORECASE)
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    req_str = row['Blending Requirements']
    for match in less_than_pat.finditer(req_str):
        g = match.group(1).strip()
        pct = float(match.group(2))
        if g not in G:
            raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{b}'")
        upper_bound[b][g] = min(upper_bound[b][g], pct / 100.0)
    for match in more_than_pat.finditer(req_str):
        g = match.group(1).strip()
        pct = float(match.group(2))
        if g not in G:
            raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{b}'")
        lower_bound[b][g] = max(lower_bound[b][g], pct / 100.0)
m = Model('wine_blending')
x_vars = m.addVars(G, B, lb=0.0, vtype=GRB.CONTINUOUS, name='')
for b in B:
    total_b = quicksum((x_vars[g, b] for g in G))
    for g in G:
        lb = lower_bound[b][g]
        ub = upper_bound[b][g]
        if lb > 0.0:
            m.addConstr(x_vars[g, b] >= lb * total_b, name=f'blend_lb_{g}_{b}')
        if ub < 1.0:
            m.addConstr(x_vars[g, b] <= ub * total_b, name=f'blend_ub_{g}_{b}')
for g in G:
    m.addConstr(quicksum((x_vars[g, b] for b in B)) <= supply_g[g], name=f'supply_{g}')
if 'Red' not in B:
    raise ValueError("Brand 'Red' not found in brands data")
m.addConstr(quicksum((x_vars[g, 'Red'] for g in G)) >= 2000, name='minprod_Red')
sales_revenue = quicksum((price_b[b] * quicksum((x_vars[g, b] for g in G)) for b in B))
raw_material_cost = quicksum((cost_g[g] * x_vars[g, b] for g in G for b in B))
net_profit = sales_revenue - raw_material_cost
m.setObjective(net_profit, GRB.MAXIMIZE)
m.optimize()