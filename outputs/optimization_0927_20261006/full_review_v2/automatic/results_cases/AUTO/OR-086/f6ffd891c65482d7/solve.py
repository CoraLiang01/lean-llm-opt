import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade'])
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands = list(brands_df['Brand'])
grade_supply = {}
grade_cost = {}
for (_, row) in grades_df.iterrows():
    g = row['Grade']
    try:
        grade_supply[g] = float(row['Daily Supply (kg)'])
        grade_cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-1.csv for grade '{g}': {e}")
brand_price = {}
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    try:
        brand_price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
blending_bounds = {b: {} for b in brands}
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    reqs = row['Blending Requirements']
    reqs = reqs.strip()
    pattern = '([^\\s]+)\\s+(less|more)\\s+than\\s+(\\d+)%'
    for match in re.finditer(pattern, reqs, flags=re.IGNORECASE):
        grade = match.group(1).strip()
        sense = match.group(2).strip().casefold()
        percent = float(match.group(3))
        if grade not in grades:
            raise ValueError(f"Blending requirement references unknown grade '{grade}' in brand '{b}'")
        if sense == 'less':
            if 'upper' in blending_bounds[b].get(grade, {}):
                raise ValueError(f"Multiple upper bounds for grade '{grade}' in brand '{b}'")
            blending_bounds[b].setdefault(grade, {})['upper'] = percent / 100.0
        elif sense == 'more':
            if 'lower' in blending_bounds[b].get(grade, {}):
                raise ValueError(f"Multiple lower bounds for grade '{grade}' in brand '{b}'")
            blending_bounds[b].setdefault(grade, {})['lower'] = percent / 100.0
        else:
            raise ValueError(f"Unknown blending requirement sense '{sense}' in brand '{b}'")
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_brand_prod = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
total_revenue = gp.quicksum((brand_price[b] * total_brand_prod[b] for b in brands))
total_cost = gp.quicksum((grade_cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for g in blending_bounds[b]:
        bounds = blending_bounds[b][g]
        if 'upper' in bounds:
            m.addConstr(x_vars[g, b] <= bounds['upper'] * total_brand_prod[b], name=f'blend_ub_{g}_{b}')
        if 'lower' in bounds:
            m.addConstr(x_vars[g, b] >= bounds['lower'] * total_brand_prod[b], name=f'blend_lb_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in 30-2.csv; cannot enforce minimum production constraint.")
m.addConstr(total_brand_prod['Red'] >= 2000, name='minprod_Red')
m.optimize()