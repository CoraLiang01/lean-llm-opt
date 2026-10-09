import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_req(req_str):
    """
    Parses a blending requirement string such as:
    'Grade I less than 60%; Grade II more than 20%'
    Returns a list of (grade, bound_type, value) tuples.
    """
    reqs = []
    if not isinstance(req_str, str) or not req_str.strip():
        return reqs
    for part in req_str.split(';'):
        part = part.strip()
        m = re.match('Grade\\s*([IVX]+)\\s*(less|more)\\s*than\\s*([0-9]+)%', part, re.IGNORECASE)
        if m:
            grade = m.group(1).strip().upper()
            bound_type = m.group(2).strip().lower()
            value = float(m.group(3))
            reqs.append((grade, bound_type, value))
    return reqs
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
if not {'Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'}.issubset(grades_df.columns):
    raise KeyError('30-1.csv missing required columns.')
grades_df['Grade'] = grades_df['Grade'].str.strip().str.upper()
grades = list(grades_df['Grade'])
grade_supply = {}
grade_cost = {}
for (_, row) in grades_df.iterrows():
    g = row['Grade']
    try:
        grade_supply[g] = float(row['Daily Supply (kg)'])
        grade_cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
if not {'Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'}.issubset(brands_df.columns):
    raise KeyError('30-2.csv missing required columns.')
brands_df['Brand'] = brands_df['Brand'].str.strip().str.title()
brands = list(brands_df['Brand'])
brand_price = {}
brand_blend_reqs = {}
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    try:
        brand_price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
    brand_blend_reqs[b] = parse_blending_req(row['Blending Requirements'])
for (b, reqs) in brand_blend_reqs.items():
    for (g, _, _) in reqs:
        if g not in grades:
            raise ValueError(f"Blending requirement for brand {b} references unknown grade '{g}'.")
x_vars = {}
m = gp.Model('WineBlending')
for g in grades:
    for b in brands:
        x_vars[g, b] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{g}_{b}')
total_prod = {}
for b in brands:
    total_prod[b] = gp.quicksum((x_vars[g, b] for g in grades))
total_revenue = gp.quicksum((brand_price[b] * total_prod[b] for b in brands))
total_cost = gp.quicksum((grade_cost[g] * x_vars[g, b] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for (g, bound_type, value) in brand_blend_reqs[b]:
        if bound_type == 'less':
            m.addConstr(x_vars[g, b] <= value / 100.0 * total_prod[b], name=f'blend_ub_{g}_{b}')
        elif bound_type == 'more':
            m.addConstr(x_vars[g, b] >= value / 100.0 * total_prod[b], name=f'blend_lb_{g}_{b}')
        else:
            raise ValueError(f"Unknown bound type '{bound_type}' in blending requirements for brand {b}.")
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
red_brand = None
for b in brands:
    if b.strip().casefold() == 'red':
        red_brand = b
        break
if red_brand is None:
    raise ValueError("No brand named 'Red' found in 30-2.csv.")
m.addConstr(total_prod[red_brand] >= 2000.0, name='min_prod_red')
m.optimize()