import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades = [g.strip() for g in grades_df['Grade']]
brands = [b.strip() for b in brands_df['Brand']]
supply_limit = {}
cost = {}
for (_, row) in grades_df.iterrows():
    g = row['Grade'].strip()
    try:
        supply_limit[g] = float(row['Daily Supply (kg)'])
        cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-1.csv for grade '{g}': {e}")
selling_price = {}
for (_, row) in brands_df.iterrows():
    b = row['Brand'].strip()
    try:
        selling_price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
blending_bounds = {b: [] for b in brands}
for (_, row) in brands_df.iterrows():
    b = row['Brand'].strip()
    req_str = row['Blending Requirements']
    pattern = '([A-Za-z0-9]+)\\s+(less than|more than)\\s+([0-9]+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        bound_type = match.group(2).strip().lower()
        percent = float(match.group(3)) / 100.0
        if bound_type == 'less than':
            blending_bounds[b].append((grade, '<', percent))
        elif bound_type == 'more than':
            blending_bounds[b].append((grade, '>', percent))
        else:
            raise ValueError(f"Unknown blending bound type '{bound_type}' in '{req_str}' for brand '{b}'.")
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
total_revenue = gp.quicksum((total_prod[b] * selling_price[b] for b in brands))
total_cost = gp.quicksum((x_vars[g, b] * cost[g] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for (g, sense, bound) in blending_bounds[b]:
        if g not in grades:
            raise ValueError(f"Blending requirement references unknown grade '{g}' for brand '{b}'.")
        if sense == '<':
            m.addConstr(x_vars[g, b] <= bound * total_prod[b], name=f'blend_ub_{g}_{b}')
        elif sense == '>':
            m.addConstr(x_vars[g, b] >= bound * total_prod[b], name=f'blend_lb_{g}_{b}')
        else:
            raise ValueError(f"Unknown sense '{sense}' in blending requirements.")
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_limit[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(total_prod['Red'] >= 2000, name='min_prod_Red')
m.optimize()