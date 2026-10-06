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
supply = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
selling_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))

def parse_blending_reqs(req_str):
    reqs = []
    pattern = '([A-Z]+)\\s+(less than|more than)\\s+(\\d+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        sense = match.group(2).strip()
        value = float(match.group(3)) / 100.0
        if sense == 'less than':
            reqs.append((grade, '<=', value))
        elif sense == 'more than':
            reqs.append((grade, '>=', value))
        else:
            raise ValueError(f'Unknown blending sense: {sense}')
    return reqs
blending_reqs = {}
for idx, row in brands_df.iterrows():
    brand = row['Brand']
    req_str = row['Blending Requirements']
    blending_reqs[brand] = parse_blending_reqs(req_str)
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((gp.quicksum((x[g, b] for g in grades)) * selling_price[b] for b in brands))
total_cost = gp.quicksum((x[g, b] * cost[g] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    reqs = blending_reqs[b]
    total_prod = gp.quicksum((x[g, b] for g in grades))
    for g_req, sense, val in reqs:
        if g_req not in grades:
            raise ValueError(f"Grade '{g_req}' in blending requirements not found in grade list.")
        if sense == '<=':
            m.addConstr(x[g_req, b] - val * total_prod <= 0, name=f'blend_ub_{g_req}_{b}')
        elif sense == '>=':
            m.addConstr(x[g_req, b] - val * total_prod >= 0, name=f'blend_lb_{g_req}_{b}')
        else:
            raise ValueError(f'Unknown sense: {sense}')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total net profit: {m.objVal:.2f} CNY')
    print('\n--- Optimal Allocation (kg of each grade to each brand) ---')
    for b in brands:
        total_b = sum((x[g, b].X for g in grades))
        print(f'\nBrand: {b} (Total produced: {total_b:.2f} kg, Selling price: {selling_price[b]:.2f} CNY/kg)')
        for g in grades:
            val = x[g, b].X
            if val > 1e-06:
                print(f'  Grade {g}: {val:.2f} kg')
    print('\n--- Raw Material Usage ---')
    for g in grades:
        used = sum((x[g, b].X for b in brands))
        print(f'Grade {g}: Used {used:.2f} kg / Supply limit {supply[g]}')
    print('\n--- Blending Proportions (for each brand) ---')
    for b in brands:
        total_b = sum((x[g, b].X for g in grades))
        if total_b > 1e-08:
            print(f'\nBrand: {b}')
            for g in grades:
                prop = x[g, b].X / total_b
                print(f'  Grade {g}: {prop * 100:.2f}%')
else:
    print(f'No optimal solution found. Status: {m.status}')