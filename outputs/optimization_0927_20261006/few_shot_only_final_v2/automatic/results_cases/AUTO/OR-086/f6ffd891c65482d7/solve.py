import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades = list(grades_df['Grade'].astype(str))
brands = list(brands_df['Brand'].astype(str))
grade_supply = {}
grade_cost = {}
for (_, row) in grades_df.iterrows():
    g = str(row['Grade'])
    try:
        grade_supply[g] = float(row['Daily Supply (kg)'])
        grade_cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
brand_price = {}
for (_, row) in brands_df.iterrows():
    b = str(row['Brand'])
    try:
        brand_price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
blending_bounds = {b: [] for b in brands}
for (_, row) in brands_df.iterrows():
    b = str(row['Brand'])
    reqs = str(row['Blending Requirements']).strip()
    if reqs == '' or reqs.lower() == 'none':
        continue
    for req in re.split(',\\s*', reqs):
        m = re.match('Grade\\s*([IVX]+)\\s*(more than|less than)\\s*([0-9.]+)%', req, re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse blending requirement: '{req}' for brand '{b}'")
        grade_label = m.group(1).strip().upper()
        if grade_label not in grades:
            raise ValueError(f"Grade '{grade_label}' in blending requirement not found in grades list {grades}")
        bound_type = m.group(2).strip().lower()
        percent = float(m.group(3))
        blending_bounds[b].append((grade_label, bound_type, percent))
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((brand_price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((grade_cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
for b in brands:
    total_b = gp.quicksum((x_vars[g, b] for g in grades))
    for (g_req, bound_type, percent) in blending_bounds[b]:
        prop = percent / 100.0
        if bound_type == 'more than':
            m.addConstr(x_vars[g_req, b] >= prop * total_b, name=f'blend_lb_{g_req}_{b}')
        elif bound_type == 'less than':
            m.addConstr(x_vars[g_req, b] <= prop * total_b, name=f'blend_ub_{g_req}_{b}')
        else:
            raise ValueError(f"Unknown bound type '{bound_type}' in blending requirements.")
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()