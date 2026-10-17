import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
df_supply = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
df_supply['Grade'] = df_supply['Grade'].astype(str).str.strip()
if set(grades) - set(df_supply['Grade']):
    raise ValueError(f"Missing grades in 30-1.csv: {set(grades) - set(df_supply['Grade'])}")
supply_dict = {}
cost_dict = {}
for g in grades:
    row = df_supply[df_supply['Grade'] == g]
    if row.empty:
        raise ValueError(f'Grade {g} not found in 30-1.csv')
    supply_val = float(row.iloc[0]['Daily Supply (kg)'])
    cost_val = float(row.iloc[0]['Cost (CNY/kg)'])
    supply_dict[g] = supply_val
    cost_dict[g] = cost_val
df_blend = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
df_blend['Brand'] = df_blend['Brand'].astype(str).str.strip()
if set(brands) - set(df_blend['Brand']):
    raise ValueError(f"Missing brands in 30-2.csv: {set(brands) - set(df_blend['Brand'])}")
price_dict = {}
for b in brands:
    row = df_blend[df_blend['Brand'] == b]
    if row.empty:
        raise ValueError(f'Brand {b} not found in 30-2.csv')
    price_val = float(row.iloc[0]['Selling Price (CNY/kg)'])
    price_dict[b] = price_val
blend_lb = {b: {} for b in brands}
blend_ub = {b: {} for b in brands}
for b in brands:
    row = df_blend[df_blend['Brand'] == b].iloc[0]
    for g in grades:
        lb_col = f'{g} Lower Bound (%)'
        if lb_col in df_blend.columns:
            val = row[lb_col].strip()
            if val != '':
                blend_lb[b][g] = float(val) / 100.0
        ub_col = f'{g} Upper Bound (%)'
        if ub_col in df_blend.columns:
            val = row[ub_col].strip()
            if val != '':
                blend_ub[b][g] = float(val) / 100.0
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_revenue = gp.quicksum((price_dict[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
total_cost = gp.quicksum((cost_dict[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
total_b = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
for b in brands:
    for g in grades:
        if g in blend_ub[b]:
            m.addConstr(x_vars[g, b] <= blend_ub[b][g] * total_b[b], name=f'blend_ub_{g}_{b}')
        if g in blend_lb[b]:
            m.addConstr(x_vars[g, b] >= blend_lb[b][g] * total_b[b], name=f'blend_lb_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_dict[g], name=f'supply_{g}')
m.addConstr(total_b['Red'] >= 2000, name='min_prod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total net profit: {m.objVal:.2f} CNY')
    print('\n--- Production Plan (kg) ---')
    for b in brands:
        total_prod = total_b[b].getValue()
        print(f'Brand {b}: {total_prod:.2f} kg')
        for g in grades:
            val = x_vars[g, b].X
            if val > 1e-06:
                print(f'  Grade {g}: {val:.2f} kg')
    print('\n--- Raw Material Usage (kg) ---')
    for g in grades:
        used = sum((x_vars[g, b].X for b in brands))
        print(f'Grade {g}: {used:.2f} kg (Supply limit: {supply_dict[g]:.2f} kg)')
else:
    print(f'No optimal solution found. Status: {m.status}')