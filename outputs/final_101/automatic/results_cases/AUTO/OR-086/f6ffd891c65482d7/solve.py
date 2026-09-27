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
grade_supply = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
grade_cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
brand_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))

def parse_blending_reqs(req_str):
    reqs = []
    tokens = re.split('\\s{2,}', req_str.strip())
    for token in tokens:
        m = re.match('([IV]+)\\s+(less|more)\\s+than\\s+(\\d+)%', token.strip(), re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            bound_type = '<' if m.group(2).lower() == 'less' else '>'
            value = float(m.group(3)) / 100.0
            reqs.append((grade, bound_type, value))
    return reqs
blending_reqs = {}
for idx, row in brands_df.iterrows():
    brand = row['Brand']
    reqs = parse_blending_reqs(row['Blending Requirements'])
    blending_reqs[brand] = reqs
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((gp.quicksum((x[g, b] for g in grades)) * brand_price[b] for b in brands))
total_cost = gp.quicksum((x[g, b] * grade_cost[g] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    reqs = blending_reqs.get(b, [])
    total_prod = gp.quicksum((x[g, b] for g in grades))
    for g_req, bound_type, value in reqs:
        if g_req not in grades:
            raise ValueError(f"Grade '{g_req}' in blending requirements for brand '{b}' not found in grades list.")
        if bound_type == '<':
            m.addConstr(x[g_req, b] <= value * total_prod, name=f'blend_ub_{g_req}_{b}')
        elif bound_type == '>':
            m.addConstr(x[g_req, b] >= value * total_prod, name=f'blend_lb_{g_req}_{b}')
        else:
            raise ValueError(f"Unknown bound_type '{bound_type}' in blending requirements.")
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Production Plan (kg of each grade in each brand) ---')
    for b in brands:
        total_b = sum((x[g, b].X for g in grades))
        print(f'\nBrand: {b} (Total produced: {total_b:.2f} kg, Selling price: {brand_price[b]:.2f} CNY/kg)')
        for g in grades:
            val = x[g, b].X
            if val > 1e-06:
                print(f'  Grade {g}: {val:.2f} kg')
    print('\n--- Raw Material Usage (kg) ---')
    for g in grades:
        used = sum((x[g, b].X for b in brands))
        print(f'Grade {g}: {used:.2f} kg (Supply limit: {grade_supply[g]})')
    print('\n--- Brand Blending Proportions ---')
    for b in brands:
        total_b = sum((x[g, b].X for g in grades))
        if total_b > 1e-06:
            print(f'\nBrand: {b}')
            for g in grades:
                prop = x[g, b].X / total_b if total_b > 0 else 0.0
                print(f'  Grade {g}: {prop * 100:.2f}%')
else:
    print(f'No optimal solution found. Status: {m.status}')