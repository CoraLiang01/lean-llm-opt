import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades = list(grades_df['Grade'].astype(str))
brands = list(brands_df['Brand'].astype(str))
supply_limit = {}
cost = {}
for (_, row) in grades_df.iterrows():
    g = str(row['Grade'])
    try:
        supply_limit[g] = float(row['Daily Supply (kg)'])
        cost[g] = float(row['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
selling_price = {}
for (_, row) in brands_df.iterrows():
    b = str(row['Brand'])
    try:
        selling_price[b] = float(row['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')

def parse_blending_reqs(req_str):
    reqs = []
    if not req_str.strip():
        return reqs
    for part in req_str.split(','):
        part = part.strip()
        m = re.match('Grade\\s*([IVX]+)\\s*([<>])\\s*([\\d.]+)%', part)
        if m:
            grade = m.group(1)
            grade = grade.replace(' ', '')
            sense = m.group(2)
            value = float(m.group(3)) / 100.0
            reqs.append((grade, sense, value))
        else:
            raise ValueError(f"Could not parse blending requirement: '{part}'")
    return reqs
blending_reqs = {}
for (_, row) in brands_df.iterrows():
    b = str(row['Brand'])
    req_str = row['Blending Requirements']
    blending_reqs[b] = parse_blending_reqs(req_str)
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {}
for b in brands:
    total_prod[b] = gp.quicksum((x_vars[g, b] for g in grades))
total_revenue = gp.quicksum((selling_price[b] * total_prod[b] for b in brands))
total_cost = gp.quicksum((cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    reqs = blending_reqs[b]
    for (g, sense, value) in reqs:
        if sense == '<':
            m.addConstr(x_vars[g, b] <= value * total_prod[b], name=f'blend_{b}_{g}_ub')
        elif sense == '>':
            m.addConstr(x_vars[g, b] >= value * total_prod[b], name=f'blend_{b}_{g}_lb')
        else:
            raise ValueError(f"Unknown sense '{sense}' in blending requirement for brand {b}")
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_limit[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list from 30-2.csv")
m.addConstr(total_prod['Red'] >= 2000.0, name='minprod_Red')
m.optimize()