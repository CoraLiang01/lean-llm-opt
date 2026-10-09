import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades_df['Grade_norm'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade_norm'])
grade_supply = {}
grade_cost = {}
for (idx, row) in grades_df.iterrows():
    g = row['Grade_norm']
    try:
        grade_supply[g] = float(row['Daily Supply (kg)'])
        grade_cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
brands_df['Brand_norm'] = brands_df['Brand'].str.strip()
brands = list(brands_df['Brand_norm'])
brand_price = {}
brand_blendreq = {}
for (idx, row) in brands_df.iterrows():
    b = row['Brand_norm']
    try:
        brand_price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
    brand_blendreq[b] = row['Blending Requirements']
blend_bounds = {b: {} for b in brands}
for b in brands:
    req = brand_blendreq[b]
    reqs = re.split('\\s{2,}', req.strip())
    for r in reqs:
        r = r.strip()
        m = re.match('^([^\\s]+)\\s+(less|more)\\s+than\\s+([0-9]+)%$', r, re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            bound_type = m.group(2).strip().casefold()
            value = float(m.group(3)) / 100.0
            if grade not in grades:
                raise ValueError(f"Grade '{grade}' in blending requirements for brand '{b}' not found in grades list.")
            if grade not in blend_bounds[b]:
                blend_bounds[b][grade] = {'min': None, 'max': None}
            if bound_type == 'less':
                blend_bounds[b][grade]['max'] = value
            elif bound_type == 'more':
                blend_bounds[b][grade]['min'] = value
            else:
                raise ValueError(f"Unknown bound type '{bound_type}' in blending requirements for brand '{b}'.")
        elif r == '':
            continue
        else:
            raise ValueError(f"Could not parse blending requirement '{r}' for brand '{b}'.")
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
total_revenue = gp.quicksum((brand_price[b] * total_prod[b] for b in brands))
total_cost = gp.quicksum((grade_cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for g in blend_bounds[b]:
        bounds = blend_bounds[b][g]
        if bounds['max'] is not None:
            m.addConstr(x_vars[g, b] <= bounds['max'] * total_prod[b], name=f'blend_max_{g}_{b}')
        if bounds['min'] is not None:
            m.addConstr(x_vars[g, b] >= bounds['min'] * total_prod[b], name=f'blend_min_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
red_brand_key = None
for b in brands:
    if b.strip().casefold() == 'red':
        red_brand_key = b
        break
if red_brand_key is None:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(total_prod[red_brand_key] >= 2000.0, name='min_prod_Red')
m.optimize()