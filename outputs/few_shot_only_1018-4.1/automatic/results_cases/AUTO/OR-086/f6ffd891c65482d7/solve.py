import gurobipy as gp
import pandas as pd
import numpy as np
import re
grade_file = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
brand_file = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
grades_df = pd.read_csv(grade_file, sep=',', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades_df['Daily Supply (kg)'] = grades_df['Daily Supply (kg)'].str.strip()
grades_df['Cost (CNY/kg)'] = grades_df['Cost (CNY/kg)'].str.strip()
brands_df = pd.read_csv(brand_file, sep=',', dtype=str, keep_default_na=False)
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands_df['Selling Price (CNY/kg)'] = brands_df['Selling Price (CNY/kg)'].str.strip()
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
supply_g = {}
cost_g = {}
for g in grades:
    row = grades_df[grades_df['Grade'].str.strip() == g]
    if row.empty:
        raise ValueError(f"Grade '{g}' not found in 30-1.csv")
    supply_g[g] = float(row.iloc[0]['Daily Supply (kg)'])
    cost_g[g] = float(row.iloc[0]['Cost (CNY/kg)'])
price_b = {}
for b in brands:
    row = brands_df[brands_df['Brand'].str.strip() == b]
    if row.empty:
        raise ValueError(f"Brand '{b}' not found in 30-2.csv")
    price_b[b] = float(row.iloc[0]['Selling Price (CNY/kg)'])
blending_reqs = {b: [] for b in brands}
for idx, row in brands_df.iterrows():
    b = row['Brand'].strip()
    for col in brands_df.columns:
        if col == 'Brand' or col == 'Selling Price (CNY/kg)':
            continue
        req = row[col]
        if not isinstance(req, str) or req.strip() == '':
            continue
        req = req.strip()
        m = re.match('^([A-Za-z0-9]+)\\s+(less than|more than)\\s+([0-9]+)%$', req)
        if not m:
            continue
        grade = m.group(1).strip()
        if grade not in grades:
            raise ValueError(f"Unknown grade '{grade}' in blending requirement for brand '{b}'")
        if m.group(2) == 'less than':
            bound_type = 'lt'
        elif m.group(2) == 'more than':
            bound_type = 'gt'
        else:
            raise ValueError(f"Unknown bound type '{m.group(2)}' in blending requirement")
        value = float(m.group(3)) / 100.0
        blending_reqs[b].append((grade, bound_type, value))
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((price_b[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((cost_g[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for g, bound_type, value in blending_reqs[b]:
        total_blend = gp.quicksum((x_vars[gg, b] for gg in grades))
        x_gb = x_vars[g, b]
        epsilon = 1e-05
        if bound_type == 'lt':
            m.addConstr(x_gb - value * total_blend <= -epsilon * total_blend, name=f'blend_{b}_{g}_lt')
        elif bound_type == 'gt':
            m.addConstr(x_gb - value * total_blend >= epsilon * total_blend, name=f'blend_{b}_{g}_gt')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_g[g], name=f'supply_{g}')
m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()