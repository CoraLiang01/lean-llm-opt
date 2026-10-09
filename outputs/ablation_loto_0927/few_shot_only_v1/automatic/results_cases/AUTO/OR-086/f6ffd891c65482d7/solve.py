import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',')
grades = list(grades_df['Grade'].astype(str))
brands = list(brands_df['Brand'].astype(str))
grade_supply = dict(zip(grades_df['Grade'].astype(str), grades_df['Daily Supply (kg)']))
grade_cost = dict(zip(grades_df['Grade'].astype(str), grades_df['Cost (CNY/kg)']))
brand_price = dict(zip(brands_df['Brand'].astype(str), brands_df['Selling Price (CNY/kg)']))

def parse_blending_req(req_str):
    reqs = []
    if pd.isnull(req_str) or not req_str.strip():
        return reqs
    for part in req_str.split(','):
        part = part.strip()
        m = re.match('^(I{1,3})\\s+(more than|less than)\\s+(\\d+)%$', part)
        if m:
            grade = m.group(1)
            bound_type = 'lower' if m.group(2) == 'more than' else 'upper'
            percent = float(m.group(3)) / 100.0
            reqs.append((grade, bound_type, percent))
        else:
            raise ValueError(f"Cannot parse blending requirement: '{part}'")
    return reqs
brand_blend_reqs = {}
for (idx, row) in brands_df.iterrows():
    brand = str(row['Brand'])
    reqs = parse_blending_req(row['Blending Requirements'])
    brand_blend_reqs[brand] = reqs
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_sales = gp.quicksum((brand_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((grade_cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades))
m.setObjective(total_sales - total_cost, gp.GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
for b in brands:
    total_prod_b = gp.quicksum((x[g, b] for g in grades))
    for (g_req, bound_type, percent) in brand_blend_reqs[b]:
        if g_req not in grades:
            raise ValueError(f"Grade '{g_req}' in blending requirements for brand '{b}' not found in grades list.")
        if bound_type == 'lower':
            m.addConstr(x[g_req, b] >= percent * total_prod_b, name=f'blend_{b}_{g_req}_lb')
        elif bound_type == 'upper':
            m.addConstr(x[g_req, b] <= percent * total_prod_b, name=f'blend_{b}_{g_req}_ub')
        else:
            raise ValueError(f"Unknown bound type '{bound_type}' in blending requirements.")
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total net profit: {m.objVal:.2f} CNY')
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
else:
    print(f'No optimal solution found. Status: {m.status}')