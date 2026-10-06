import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',')
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
grade_map = {str(g).strip(): g for g in grades_df['Grade']}
brand_map = {str(b).strip(): b for b in brands_df['Brand']}
supply = {}
cost = {}
for g in grades:
    row = grades_df[grades_df['Grade'].astype(str).str.strip() == g]
    if row.empty:
        raise ValueError(f"Grade '{g}' not found in 30-1.csv")
    supply[g] = float(row['Daily Supply (kg)'].values[0])
    cost[g] = float(row['Cost (CNY/kg)'].values[0])
price = {}
for b in brands:
    row = brands_df[brands_df['Brand'].astype(str).str.strip() == b]
    if row.empty:
        raise ValueError(f"Brand '{b}' not found in 30-2.csv")
    price[b] = float(row['Selling Price (CNY/kg)'].values[0])
blending_reqs = {}
for b in brands:
    row = brands_df[brands_df['Brand'].astype(str).str.strip() == b]
    req_str = row['Blending Requirements'].values[0]
    reqs = []
    pattern = '([I]{1,3})\\s+(less than|more than)\\s+(\\d+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        sense = match.group(2).strip()
        value = float(match.group(3)) / 100.0
        reqs.append((grade, sense, value))
    blending_reqs[b] = reqs
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {b: gp.quicksum((x[g, b] for g in grades)) for b in brands}
total_revenue = gp.quicksum((price[b] * total_prod[b] for b in brands))
total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for g_req, sense, val in blending_reqs[b]:
        if sense == 'less than':
            m.addConstr(x[g_req, b] <= val * total_prod[b], name=f'blend_ub_{g_req}_{b}')
        elif sense == 'more than':
            m.addConstr(x[g_req, b] >= val * total_prod[b], name=f'blend_lb_{g_req}_{b}')
        else:
            raise ValueError(f"Unknown blending sense '{sense}' in blending requirements for brand '{b}'.")
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
m.addConstr(total_prod['Red'] >= 2000, name='min_prod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total net profit: {m.objVal:.2f} CNY')
    print('\n--- Production Plan (kg of each grade in each brand) ---')
    for b in brands:
        print(f'Brand: {b}')
        for g in grades:
            val = x[g, b].X
            if val > 1e-06:
                print(f'  Grade {g}: {val:.2f} kg')
        print(f'  Total produced: {total_prod[b].getValue():.2f} kg')
    print('\n--- Raw Material Usage (kg) ---')
    for g in grades:
        used = sum((x[g, b].X for b in brands))
        print(f'Grade {g}: {used:.2f} kg (Supply limit: {supply[g]:.2f} kg)')
else:
    print(f'No optimal solution found. Status: {m.status}')