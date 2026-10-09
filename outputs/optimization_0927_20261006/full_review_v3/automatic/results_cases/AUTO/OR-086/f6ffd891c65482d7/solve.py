import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str, grades_set):
    """
    Parses a blending requirements string like:
    'I less than 10%  II more than 50%'
    Returns: dict of {grade: (lower, upper)} where lower/upper are floats in [0,1] or None if not specified.
    """
    bounds = {g: [None, None] for g in grades_set}
    pattern = '([^\\s]+)\\s+(less than|more than)\\s+([0-9]+)%'
    for match in re.finditer(pattern, req_str, flags=re.IGNORECASE):
        grade = match.group(1).strip()
        op = match.group(2).strip().casefold()
        percent = float(match.group(3))
        if grade not in grades_set:
            continue
        if op == 'less than':
            bounds[grade][1] = percent / 100.0
        elif op == 'more than':
            bounds[grade][0] = percent / 100.0
    return {g: (bounds[g][0], bounds[g][1]) for g in grades_set}
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade'])
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands = list(brands_df['Brand'])
supply_limit = {}
cost = {}
for (_, row) in grades_df.iterrows():
    g = row['Grade']
    supply_limit[g] = float(row['Daily Supply (kg)'])
    cost[g] = float(row['Cost (CNY/kg)'])
selling_price = {}
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    selling_price[b] = float(row['Selling Price (CNY/kg)'])
blending_bounds = {b: {} for b in brands}
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    req_str = row['Blending Requirements']
    bounds = parse_blending_requirements(req_str, set(grades))
    blending_bounds[b] = bounds
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
revenue_expr = gp.quicksum((selling_price[b] * total_prod[b] for b in brands))
cost_expr = gp.quicksum((cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(revenue_expr - cost_expr, gp.GRB.MAXIMIZE)
for b in brands:
    for g in grades:
        (lower, upper) = blending_bounds[b][g]
        if lower is not None:
            m.addConstr(x_vars[g, b] >= lower * total_prod[b], name=f'blend_lb_{g}_{b}')
        if upper is not None:
            m.addConstr(x_vars[g, b] <= upper * total_prod[b], name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_limit[g], name=f'supply_{g}')
if 'Red' in brands:
    m.addConstr(total_prod['Red'] >= 2000, name='min_prod_Red')
else:
    raise ValueError("Brand 'Red' not found in brands list.")
m.optimize()