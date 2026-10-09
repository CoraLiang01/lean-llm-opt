import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parses a blending requirements string like:
    'I less than 10%  II more than 50%'
    Returns a list of (grade, bound_type, value) tuples.
    bound_type: 'lt' for less than, 'gt' for more than
    value: float (proportion, e.g., 0.10)
    """
    reqs = []
    tokens = re.split('\\s{2,}', req_str.strip())
    for token in tokens:
        m = re.match('([^\\s]+)\\s+(less|more)\\s+than\\s+([0-9.]+)%', token.strip(), re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            bound_type = 'lt' if m.group(2).strip().casefold() == 'less' else 'gt'
            value = float(m.group(3)) / 100.0
            reqs.append((grade, bound_type, value))
    return reqs
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades = list(grades_df['Grade'])
supply = {}
cost = {}
for (_, row) in grades_df.iterrows():
    g = row['Grade']
    try:
        supply[g] = float(row['Daily Supply (kg)'])
        cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands = list(brands_df['Brand'])
price = {}
blending_reqs = {}
for (_, row) in brands_df.iterrows():
    b = row['Brand']
    try:
        price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
    blending_reqs[b] = parse_blending_requirements(row['Blending Requirements'])
if set(grades) != set(['I', 'II', 'III']):
    raise ValueError(f'Expected grades I, II, III, got {grades}')
if set(brands) != set(['Red', 'Yellow', 'Blue']):
    raise ValueError(f'Expected brands Red, Yellow, Blue, got {brands}')
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
total_revenue = gp.quicksum((price[b] * total_prod[b] for b in brands))
total_cost = gp.quicksum((cost[g] * x_vars[g, b] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    reqs = blending_reqs[b]
    for (g_req, bound_type, value) in reqs:
        if g_req not in grades:
            raise ValueError(f"Unknown grade '{g_req}' in blending requirements for brand '{b}'")
        if bound_type == 'lt':
            m.addConstr(x_vars[g_req, b] <= value * total_prod[b], name=f'blend_{b}_{g_req}_lt')
        elif bound_type == 'gt':
            m.addConstr(x_vars[g_req, b] >= value * total_prod[b], name=f'blend_{b}_{g_req}_gt')
        else:
            raise ValueError(f"Unknown bound_type '{bound_type}' in blending requirements for brand '{b}'")
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
m.addConstr(total_prod['Red'] >= 2000, name='min_prod_Red')
m.optimize()