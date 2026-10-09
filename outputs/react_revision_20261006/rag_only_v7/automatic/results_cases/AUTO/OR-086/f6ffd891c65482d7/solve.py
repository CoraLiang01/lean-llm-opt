import pandas as pd
import numpy as np
import re
import gurobipy as gp
from gurobipy import GRB
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
brands_df['Brand'] = brands_df['Brand'].str.strip()
grades = list(grades_df['Grade'])
brands = list(brands_df['Brand'])
try:
    supply_limit = {row['Grade']: int(row['Daily Supply (kg)']) for (_, row) in grades_df.iterrows()}
    cost = {row['Grade']: float(row['Cost (CNY/kg)']) for (_, row) in grades_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing 30-1.csv: {e}')
try:
    price = {row['Brand']: float(row['Selling Price (CNY/kg)']) for (_, row) in brands_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing 30-2.csv: {e}')
blending_bounds = {b: {} for b in brands}
for (_, row) in brands_df.iterrows():
    brand = row['Brand']
    reqs = row['Blending Requirements']
    reqs_split = re.split('\\s{2,}|\\t', reqs)
    for req in reqs_split:
        req = req.strip()
        m_less = re.match('^([A-Za-z0-9]+)\\s+less\\s+than\\s+([0-9]+)%$', req, re.IGNORECASE)
        m_more = re.match('^([A-Za-z0-9]+)\\s+more\\s+than\\s+([0-9]+)%$', req, re.IGNORECASE)
        if m_less:
            g = m_less.group(1).strip()
            ub = float(m_less.group(2)) / 100.0
            if g not in grades:
                raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{brand}'")
            if g not in blending_bounds[brand]:
                blending_bounds[brand][g] = {}
            blending_bounds[brand][g]['ub'] = ub
        elif m_more:
            g = m_more.group(1).strip()
            lb = float(m_more.group(2)) / 100.0
            if g not in grades:
                raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{brand}'")
            if g not in blending_bounds[brand]:
                blending_bounds[brand][g] = {}
            blending_bounds[brand][g]['lb'] = lb
        elif req == '':
            continue
        else:
            raise ValueError(f"Unrecognized blending requirement: '{req}' for brand '{brand}'")
for g in grades:
    if g not in supply_limit or g not in cost:
        raise ValueError(f"Missing supply or cost for grade '{g}'")
for b in brands:
    if b not in price:
        raise ValueError(f"Missing price for brand '{b}'")
m = gp.Model('wine_blending')
m.Params.MIPGap = 0.0001
quantity_vars = m.addVars([(g, b) for g in grades for b in brands], lb=0.0, vtype=GRB.CONTINUOUS, name='')
total_prod = {}
for b in brands:
    total_prod[b] = gp.quicksum((quantity_vars[g, b] for g in grades))
revenue = gp.quicksum((total_prod[b] * price[b] for b in brands))
raw_cost = gp.quicksum((quantity_vars[g, b] * cost[g] for g in grades for b in brands))
m.setObjective(revenue - raw_cost, GRB.MAXIMIZE)
for b in brands:
    for g in blending_bounds[b]:
        bounds = blending_bounds[b][g]
        if 'lb' in bounds:
            m.addConstr(quantity_vars[g, b] - bounds['lb'] * total_prod[b] >= 0, name='')
        if 'ub' in bounds:
            m.addConstr(quantity_vars[g, b] - bounds['ub'] * total_prod[b] <= 0, name='')
for g in grades:
    m.addConstr(gp.quicksum((quantity_vars[g, b] for b in brands)) <= supply_limit[g], name='')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list")
m.addConstr(total_prod['Red'] >= 2000, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for g in grades:
        for b in brands:
            var = quantity_vars[g, b]
            print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')