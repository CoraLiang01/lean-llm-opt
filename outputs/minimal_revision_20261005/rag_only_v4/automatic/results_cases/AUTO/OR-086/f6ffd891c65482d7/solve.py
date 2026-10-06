import pandas as pd
import numpy as np
import re
import gurobipy as gp
from gurobipy import GRB
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',')
grades_df['Grade'] = grades_df['Grade'].astype(str).str.strip()
brands_df['Brand'] = brands_df['Brand'].astype(str).str.strip()
grades = list(grades_df['Grade'])
brands = list(brands_df['Brand'])
supply = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
selling_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))

def parse_blending_req(req_str):
    reqs = {}
    parts = re.split('\\s{2,}', req_str.strip())
    for part in parts:
        m1 = re.match('([IV]+)\\s+less\\s+than\\s+(\\d+)%', part, re.IGNORECASE)
        if m1:
            grade = m1.group(1).strip()
            upper = float(m1.group(2)) / 100.0
            if grade not in reqs:
                reqs[grade] = [None, None]
            reqs[grade][1] = upper
            continue
        m2 = re.match('([IV]+)\\s+more\\s+than\\s+(\\d+)%', part, re.IGNORECASE)
        if m2:
            grade = m2.group(1).strip()
            lower = float(m2.group(2)) / 100.0
            if grade not in reqs:
                reqs[grade] = [None, None]
            reqs[grade][0] = lower
            continue
    for k in reqs:
        reqs[k] = tuple(reqs[k])
    return reqs
blending_reqs = {}
for (idx, row) in brands_df.iterrows():
    brand = row['Brand']
    req_str = row['Blending Requirements']
    blending_reqs[brand] = parse_blending_req(req_str)
if set(grades) != set(grades_df['Grade']):
    raise ValueError('Mismatch in grades between index set and data.')
if set(brands) != set(brands_df['Brand']):
    raise ValueError('Mismatch in brands between index set and data.')
for g in grades:
    if g not in supply or g not in cost:
        raise ValueError(f'Missing supply or cost for grade {g}.')
for b in brands:
    if b not in selling_price:
        raise ValueError(f'Missing selling price for brand {b}.')
m = gp.Model('wine_blending')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
total_prod = {}
for b in brands:
    total_prod[b] = gp.quicksum((x[g, b] for g in grades))
sales_revenue = gp.quicksum((selling_price[b] * total_prod[b] for b in brands))
raw_material_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
m.setObjective(sales_revenue - raw_material_cost, GRB.MAXIMIZE)
for b in brands:
    reqs = blending_reqs.get(b, {})
    for g in grades:
        bounds = reqs.get(g, (None, None))
        (lower, upper) = bounds
        if lower is not None:
            m.addConstr(x[g, b] >= lower * total_prod[b], name='')
        if upper is not None:
            m.addConstr(x[g, b] <= upper * total_prod[b], name='')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name='')
m.addConstr(total_prod['Red'] >= 2000, name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for g in grades:
        for b in brands:
            var = x[g, b]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')