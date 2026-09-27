import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',')
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']

def norm(x):
    return str(x).strip()
grades_df['Grade'] = grades_df['Grade'].apply(norm)
brands_df['Brand'] = brands_df['Brand'].apply(norm)
supply = {}
cost = {}
for g in grades:
    row = grades_df[grades_df['Grade'] == g]
    if row.empty:
        raise ValueError(f"Grade '{g}' not found in 30-1.csv")
    supply[g] = float(row['Daily Supply (kg)'].iloc[0])
    cost[g] = float(row['Cost (CNY/kg)'].iloc[0])
selling_price = {}
for b in brands:
    row = brands_df[brands_df['Brand'] == b]
    if row.empty:
        raise ValueError(f"Brand '{b}' not found in 30-2.csv")
    selling_price[b] = float(row['Selling Price (CNY/kg)'].iloc[0])
blending_bounds = {b: {g: [None, None] for g in grades} for b in brands}
for b in brands:
    req_str = brands_df.loc[brands_df['Brand'] == b, 'Blending Requirements'].values[0]
    reqs = re.split('\\s{2,}', req_str.strip())
    if len(reqs) == 1:
        reqs = re.split('\\s{1,}', req_str.strip())
    for req in reqs:
        req = req.strip()
        m = re.match('([I]{1,3})\\s+(less|more)\\s+than\\s+(\\d+)%', req, re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            bound_type = m.group(2).strip().lower()
            percent = float(m.group(3))
            prop = percent / 100.0
            if grade not in grades:
                raise ValueError(f"Unknown grade '{grade}' in blending requirements for brand '{b}'")
            if bound_type == 'less':
                blending_bounds[b][grade][1] = prop
            elif bound_type == 'more':
                blending_bounds[b][grade][0] = prop
            else:
                raise ValueError(f"Unknown bound type '{bound_type}' in blending requirements for brand '{b}'")
        else:
            continue
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
for b in brands:
    total_b = gp.quicksum((x[g, b] for g in grades))
    for g in grades:
        lower, upper = blending_bounds[b][g]
        if lower is not None:
            m.addConstr(x[g, b] >= lower * total_b, name=f'blend_lb_{g}_{b}')
        if upper is not None:
            m.addConstr(x[g, b] <= upper * total_b, name=f'blend_ub_{g}_{b}')
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total net profit: {m.objVal:.2f} CNY')
    print('\n--- Optimal Blending Plan (kg) ---')
    for b in brands:
        total_b = sum((x[g, b].X for g in grades))
        print(f'\nBrand {b}: Total produced = {total_b:.2f} kg')
        for g in grades:
            print(f'  Grade {g}: {x[g, b].X:.2f} kg')
    print('\n--- Raw Material Usage (kg) ---')
    for g in grades:
        used = sum((x[g, b].X for b in brands))
        print(f'Grade {g}: {used:.2f} kg (Supply limit: {supply[g]})')
else:
    print(f'No optimal solution found. Status: {m.status}')