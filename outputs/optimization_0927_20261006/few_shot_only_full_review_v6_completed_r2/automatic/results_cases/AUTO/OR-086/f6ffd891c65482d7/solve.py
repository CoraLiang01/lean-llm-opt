import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_req(req_str, grades_set):
    """
    Parses a blending requirement string like:
    "I: 30%-50%; II: 20%-; III: -40%"
    Returns: dict {grade: (lower, upper)} with bounds as floats in [0,1] or None if not specified
    """
    reqs = {}
    for part in req_str.split(';'):
        part = part.strip()
        if not part:
            continue
        m = re.match('^([^\\s:]+)\\s*:\\s*([0-9]*)%\\s*-\\s*([0-9]*)%?$', part)
        if m:
            grade = m.group(1).strip()
            lower = m.group(2)
            upper = m.group(3)
            lower_val = float(lower) / 100 if lower else None
            upper_val = float(upper) / 100 if upper else None
            reqs[grade] = (lower_val, upper_val)
        else:
            m2 = re.match('^([^\\s:]+)\\s*:\\s*([0-9]*)%\\s*-\\s*([0-9]*)%$', part)
            if m2:
                grade = m2.group(1).strip()
                lower = m2.group(2)
                upper = m2.group(3)
                lower_val = float(lower) / 100 if lower else None
                upper_val = float(upper) / 100 if upper else None
                reqs[grade] = (lower_val, upper_val)
            else:
                m3 = re.match('^([^\\s:]+)\\s*:\\s*([0-9]*)%-$', part)
                if m3:
                    grade = m3.group(1).strip()
                    lower = m3.group(2)
                    lower_val = float(lower) / 100 if lower else None
                    reqs[grade] = (lower_val, None)
                else:
                    m4 = re.match('^([^\\s:]+)\\s*:\\s*-(\\d+)%$', part)
                    if m4:
                        grade = m4.group(1).strip()
                        upper = m4.group(2)
                        upper_val = float(upper) / 100 if upper else None
                        reqs[grade] = (None, upper_val)
    return {g: reqs[g] for g in reqs if g in grades_set}
grades_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
brands_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
grades_df = pd.read_csv(grades_path, dtype=str, keep_default_na=False)
brands_df = pd.read_csv(brands_path, dtype=str, keep_default_na=False)
grades = grades_df['Grade'].str.strip().tolist()
grades_set = set(grades)
brands = brands_df['Brand'].str.strip().tolist()
brands_set = set(brands)
supply_g = {}
cost_g = {}
for (idx, row) in grades_df.iterrows():
    g = row['Grade'].strip()
    try:
        supply_g[g] = float(row['Daily Supply (kg)'])
        cost_g[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-1.csv for grade '{g}': {e}")
price_b = {}
blending_req_b = {}
for (idx, row) in brands_df.iterrows():
    b = row['Brand'].strip()
    try:
        price_b[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
    blending_req_b[b] = parse_blending_req(row['Blending Requirements'], grades_set)
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((price_b[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((cost_g[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    total_b = gp.quicksum((x_vars[g, b] for g in grades))
    for g in blending_req_b[b]:
        (lower, upper) = blending_req_b[b][g]
        if lower is not None:
            m.addConstr(x_vars[g, b] >= lower * total_b, name=f'blend_lb_{g}_{b}')
        if upper is not None:
            m.addConstr(x_vars[g, b] <= upper * total_b, name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_g[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in 30-2.csv")
m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()