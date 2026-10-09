import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str, grades_set):
    """
    Parses a blending requirements string like:
    'I less than 10%  II more than 50%'
    Returns: dict of {grade: {'upper': float or None, 'lower': float or None}}
    All percentages are returned as fractions (e.g., 0.1 for 10%)
    """
    reqs = {g: {'upper': None, 'lower': None} for g in grades_set}
    pattern = '([^\\s]+)\\s+(less than|more than)\\s+([0-9]+)%'
    for match in re.finditer(pattern, req_str, flags=re.IGNORECASE):
        grade = match.group(1).strip()
        op = match.group(2).strip().casefold()
        pct = float(match.group(3)) / 100.0
        if grade not in grades_set:
            raise ValueError(f"Unknown grade '{grade}' in blending requirements: '{req_str}'")
        if op == 'less than':
            reqs[grade]['upper'] = pct
        elif op == 'more than':
            reqs[grade]['lower'] = pct
        else:
            raise ValueError(f"Unknown operator '{op}' in blending requirements: '{req_str}'")
    return reqs
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade'])
grades_set = set(grades)
grade_supply = {}
grade_cost = {}
for (idx, row) in grades_df.iterrows():
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
brand_blend_reqs = {}
for (idx, row) in brands_df.iterrows():
    b = row['Brand']
    try:
        price = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
    brand_price[b] = price
    req_str = row['Blending Requirements']
    blend_reqs = parse_blending_requirements(req_str, grades_set)
    brand_blend_reqs[b] = blend_reqs
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((brand_price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((grade_cost[g] * x_vars[g, b] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    total_prod = gp.quicksum((x_vars[g, b] for g in grades))
    for g in grades:
        req = brand_blend_reqs[b][g]
        if req['upper'] is not None:
            m.addConstr(x_vars[g, b] <= req['upper'] * total_prod, name=f'blend_ub_{g}_{b}')
        if req['lower'] is not None:
            m.addConstr(x_vars[g, b] >= req['lower'] * total_prod, name=f'blend_lb_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
if 'Red' not in brands_set:
    raise ValueError("Brand 'Red' not found in 30-2.csv; cannot enforce minimum production constraint.")
m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000, name='min_prod_Red')
m.optimize()