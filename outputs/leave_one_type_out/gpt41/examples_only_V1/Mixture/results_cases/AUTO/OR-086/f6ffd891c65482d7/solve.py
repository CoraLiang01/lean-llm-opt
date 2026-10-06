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
supply_limit = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
selling_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))

def parse_blending_req(req_str):
    reqs = {}
    pattern = '([A-Za-z0-9]+)\\s+(less than|more than)\\s+([0-9]+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        bound_type = match.group(2).strip().lower()
        value = float(match.group(3)) / 100.0
        if grade not in reqs:
            reqs[grade] = [None, None]
        if bound_type == 'less than':
            reqs[grade][1] = value
        elif bound_type == 'more than':
            reqs[grade][0] = value
    return reqs
blending_reqs = {}
for idx, row in brands_df.iterrows():
    brand = row['Brand']
    reqs = parse_blending_req(row['Blending Requirements'])
    full_reqs = {}
    for g in grades:
        if g in reqs:
            full_reqs[g] = tuple(reqs[g])
        else:
            full_reqs[g] = (None, None)
    blending_reqs[brand] = full_reqs
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((gp.quicksum((x[g, b] for g in grades)) * selling_price[b] for b in brands))
total_cost = gp.quicksum((x[g, b] * cost[g] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    total_b = gp.quicksum((x[g, b] for g in grades))
    for g in grades:
        lower, upper = blending_reqs[b][g]
        if lower is not None:
            m.addConstr(x[g, b] >= lower * total_b, name=f'blend_lb_{g}_{b}')
        if upper is not None:
            m.addConstr(x[g, b] <= upper * total_b, name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply_limit[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total net profit: {m.objVal:.2f} CNY')
    print('\n--- Optimal Allocation (kg) ---')
    for b in brands:
        total_b = sum((x[g, b].X for g in grades))
        print(f'\nBrand: {b} (Total produced: {total_b:.2f} kg, Selling price: {selling_price[b]:.2f} CNY/kg)')
        for g in grades:
            val = x[g, b].X
            if val > 1e-06:
                print(f'  Grade {g}: {val:.2f} kg')
        if total_b > 1e-06:
            print('  Proportions:')
            for g in grades:
                prop = x[g, b].X / total_b
                print(f'    {g}: {prop * 100:.2f}%')
else:
    print(f'No optimal solution found. Status: {m.status}')