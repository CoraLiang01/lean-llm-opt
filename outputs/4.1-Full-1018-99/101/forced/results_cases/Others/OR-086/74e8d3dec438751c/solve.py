import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str, grades_set):
    """
    Parses a blending requirements string into a dict of {grade: (lower, upper)}.
    Lower and upper are floats in [0,1], or None if not specified.
    """
    reqs = {}
    req_str = re.sub('\\s+', ' ', req_str.strip())
    pattern = '(\\b\\w+\\b)\\s*(less than|more than|<|>)\\s*([\\d.]+)%'
    for match in re.finditer(pattern, req_str, flags=re.IGNORECASE):
        grade, op, val = match.groups()
        grade = grade.strip()
        op = op.strip().casefold()
        val = float(val) / 100.0
        if grade not in grades_set:
            continue
        if grade not in reqs:
            reqs[grade] = [None, None]
        if op in ('less than', '<'):
            reqs[grade][1] = val
        elif op in ('more than', '>'):
            reqs[grade][0] = val
    return {g: (l, u) for g, (l, u) in reqs.items()}
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade'])
grades_set = set(grades)
grade_supply = {}
grade_cost = {}
for idx, row in grades_df.iterrows():
    g = row['Grade']
    try:
        supply = int(row['Daily Supply (kg)'])
        cost = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-1.csv for grade '{g}': {e}")
    grade_supply[g] = supply
    grade_cost[g] = cost
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands = list(brands_df['Brand'])
brands_set = set(brands)
brand_price = {}
brand_blend_bounds = {}
for idx, row in brands_df.iterrows():
    b = row['Brand']
    try:
        price = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
    brand_price[b] = price
    blend_req = row['Blending Requirements']
    bounds = parse_blending_requirements(blend_req, grades_set)
    brand_blend_bounds[b] = bounds
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
revenue = gp.quicksum((brand_price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
cost = gp.quicksum((grade_cost[g] * x_vars[g, b] for g in grades for b in brands))
m.setObjective(revenue - cost, gp.GRB.MAXIMIZE)
for b in brands:
    bounds = brand_blend_bounds.get(b, {})
    total_prod = gp.quicksum((x_vars[g, b] for g in grades))
    for g in grades:
        if g in bounds:
            lower, upper = bounds[g]
            if lower is not None:
                m.addConstr(x_vars[g, b] >= lower * total_prod, name=f'blend_lb_{g}_{b}')
            if upper is not None:
                m.addConstr(x_vars[g, b] <= upper * total_prod, name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in 30-2.csv; cannot enforce minimum production constraint.")
m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Production Plan (kg of each grade in each brand) ---')
    for b in brands:
        total_b = sum((x_vars[g, b].X for g in grades))
        print(f'Brand {b}: total {total_b:.2f} kg')
        for g in grades:
            val = x_vars[g, b].X
            if val > 1e-06:
                print(f'  Grade {g}: {val:.2f} kg')
    print('\n--- Raw Material Usage (kg) ---')
    for g in grades:
        used = sum((x_vars[g, b].X for b in brands))
        print(f'Grade {g}: used {used:.2f} kg (supply limit {grade_supply[g]})')
else:
    print(f'No optimal solution found. Status: {m.status}')