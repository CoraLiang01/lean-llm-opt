import pandas as pd
import numpy as np
import re
from gurobipy import Model, GRB, quicksum
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade'])
supply_g = {}
cost_g = {}
for (idx, row) in grades_df.iterrows():
    g = row['Grade']
    try:
        supply_g[g] = int(row['Daily Supply (kg)'])
        cost_g[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', dtype=str, keep_default_na=False)
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands = list(brands_df['Brand'])
price_b = {}
bounds = {g: {b: (None, None) for b in brands} for g in grades}
for (idx, row) in brands_df.iterrows():
    b = row['Brand']
    try:
        price_b[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid selling price for brand {b}: {e}')
    reqs = row['Blending Requirements']
    reqs = reqs.strip()
    pattern = '([^\\s]+)\\s+(less|more)\\s+than\\s+(\\d+)%'
    for match in re.finditer(pattern, reqs, flags=re.IGNORECASE):
        grade = match.group(1).strip()
        op = match.group(2).strip().casefold()
        percent = float(match.group(3))
        if grade not in grades:
            raise ValueError(f"Unknown grade '{grade}' in blending requirements for brand '{b}'")
        (lower, upper) = bounds[grade][b]
        if op == 'less':
            upper = percent / 100.0
        elif op == 'more':
            lower = percent / 100.0
        else:
            raise ValueError(f"Unknown operator '{op}' in blending requirements for brand '{b}'")
        bounds[grade][b] = (lower, upper)
if set(grades) != set(grades_df['Grade']):
    raise ValueError('Mismatch in grades between data and parsed set.')
if set(brands) != set(brands_df['Brand']):
    raise ValueError('Mismatch in brands between data and parsed set.')
m = Model('wine_blending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
sales_revenue = quicksum((price_b[b] * quicksum((x_vars[g, b] for g in grades)) for b in brands))
raw_material_cost = quicksum((cost_g[g] * x_vars[g, b] for g in grades for b in brands))
m.setObjective(sales_revenue - raw_material_cost, GRB.MAXIMIZE)
for b in brands:
    total_prod_b = quicksum((x_vars[g, b] for g in grades))
    for g in grades:
        (lower, upper) = bounds[g][b]
        if lower is not None:
            m.addConstr(x_vars[g, b] >= lower * total_prod_b, name=f'blend_lb_{g}_{b}')
        if upper is not None:
            m.addConstr(x_vars[g, b] <= upper * total_prod_b, name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(quicksum((x_vars[g, b] for b in brands)) <= supply_g[g], name=f'supply_{g}')
red_brand = None
for b in brands:
    if b.strip().casefold() == 'red':
        red_brand = b
        break
if red_brand is None:
    raise ValueError("No brand named 'Red' found in brands list.")
m.addConstr(quicksum((x_vars[g, red_brand] for g in grades)) >= 2000, name='minprod_red')
m.optimize()