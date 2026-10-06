import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',')
grades_df['Grade'] = grades_df['Grade'].astype(str).str.strip()
brands_df['Brand'] = brands_df['Brand'].astype(str).str.strip()
grades = list(grades_df['Grade'])
brands = list(brands_df['Brand'])
grade_supply = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
grade_cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
brand_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))

def parse_blending_req(req_str):
    reqs = {}
    pattern = '([A-Za-z0-9]+)\\s+(less than|more than)\\s+([0-9]+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        bound_type = match.group(2).strip()
        percent = float(match.group(3)) / 100.0
        if grade not in reqs:
            reqs[grade] = [None, None]
        if bound_type == 'less than':
            reqs[grade][1] = percent
        elif bound_type == 'more than':
            reqs[grade][0] = percent
    return reqs
brand_blend_bounds = {}
for idx, row in brands_df.iterrows():
    brand = row['Brand']
    req_str = row['Blending Requirements']
    bounds = parse_blending_req(req_str)
    bounds = {g: (b[0], b[1]) for g, b in bounds.items()}
    brand_blend_bounds[brand] = bounds
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
sales_revenue = gp.quicksum((brand_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
raw_cost = gp.quicksum((grade_cost[g] * x[g, b] for g in grades for b in brands))
m.setObjective(sales_revenue - raw_cost, gp.GRB.MAXIMIZE)
for b in brands:
    total_prod = gp.quicksum((x[g, b] for g in grades))
    bounds = brand_blend_bounds.get(b, {})
    for g in grades:
        if g in bounds:
            lower, upper = bounds[g]
            if lower is not None:
                m.addConstr(x[g, b] >= lower * total_prod, name=f'blend_lb_{g}_{b}')
            if upper is not None:
                m.addConstr(x[g, b] <= upper * total_prod, name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
if 'Red' in brands:
    m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
else:
    raise ValueError("Brand 'Red' not found in brands list.")
m.optimize()