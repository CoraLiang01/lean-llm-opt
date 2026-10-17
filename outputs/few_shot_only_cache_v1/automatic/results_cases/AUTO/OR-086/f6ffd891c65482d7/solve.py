import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
brands_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
grades_df = pd.read_csv(grades_path, sep=',', dtype=str, keep_default_na=False)
brands_df = pd.read_csv(brands_path, sep=',', dtype=str, keep_default_na=False)
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
if not set(grades).issubset(set(grades_df['Grade'])):
    raise ValueError(f"Missing required grades in 30-1.csv: {set(grades) - set(grades_df['Grade'])}")
grade_supply = {}
grade_cost = {}
for g in grades:
    row = grades_df.loc[grades_df['Grade'] == g]
    if row.empty:
        raise ValueError(f'Grade {g} not found in 30-1.csv')
    supply_val = row['Daily Supply (kg)'].iloc[0]
    cost_val = row['Cost (CNY/kg)'].iloc[0]
    try:
        grade_supply[g] = float(supply_val)
        grade_cost[g] = float(cost_val)
    except Exception:
        raise ValueError(f'Non-numeric supply or cost for grade {g}: {supply_val}, {cost_val}')
if not set(brands).issubset(set(brands_df['Brand'])):
    raise ValueError(f"Missing required brands in 30-2.csv: {set(brands) - set(brands_df['Brand'])}")
brand_price = {}
brand_blend_req = {}
for b in brands:
    row = brands_df.loc[brands_df['Brand'] == b]
    if row.empty:
        raise ValueError(f'Brand {b} not found in 30-2.csv')
    price_val = row['Selling Price (CNY/kg)'].iloc[0]
    blend_val = row['Blending Requirements'].iloc[0]
    try:
        brand_price[b] = float(price_val)
    except Exception:
        raise ValueError(f'Non-numeric selling price for brand {b}: {price_val}')
    brand_blend_req[b] = blend_val
blend_constraints = {b: [] for b in brands}
for b in brands:
    req = brand_blend_req[b]
    pattern = '([I]{1,3}|II|III)\\s+(less than|more than|less than or equal to|more than or equal to)\\s+([0-9]+)%'
    for match in re.finditer(pattern, req):
        g, sense_str, pct = match.groups()
        g = g.strip()
        pct_val = float(pct) / 100.0
        if sense_str == 'less than':
            sense = 'le'
            bound = pct_val
        elif sense_str == 'more than':
            sense = 'ge'
            bound = pct_val
        elif sense_str == 'less than or equal to':
            sense = 'le'
            bound = pct_val
        elif sense_str == 'more than or equal to':
            sense = 'ge'
            bound = pct_val
        else:
            raise ValueError(f'Unknown blending requirement sense: {sense_str}')
        blend_constraints[b].append((g, sense, bound))
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_prod = {b: gp.quicksum((x_vars[g, b] for g in grades)) for b in brands}
total_revenue = gp.quicksum((brand_price[b] * total_prod[b] for b in brands))
total_cost = gp.quicksum((grade_cost[g] * x_vars[g, b] for g in grades for b in brands))
m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
for b in brands:
    for g, sense, bound in blend_constraints[b]:
        if sense == 'le':
            m.addConstr(x_vars[g, b] - bound * total_prod[b] <= 0, name=f'blend_{b}_{g}_le_{bound:.2f}')
        elif sense == 'ge':
            m.addConstr(x_vars[g, b] - bound * total_prod[b] >= 0, name=f'blend_{b}_{g}_ge_{bound:.2f}')
        else:
            raise ValueError(f'Unknown sense {sense} in blending constraints.')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
m.addConstr(total_prod['Red'] >= 2000, name='min_prod_Red')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Production Plan (kg) ---')
    for b in brands:
        prod = total_prod[b].getValue()
        print(f'Brand {b}: {prod:.2f} kg')
        for g in grades:
            val = x_vars[g, b].X
            if val > 1e-06:
                print(f'  Uses Grade {g}: {val:.2f} kg')
    print('\n--- Raw Material Usage (kg) ---')
    for g in grades:
        used = sum((x_vars[g, b].X for b in brands))
        print(f'Grade {g}: {used:.2f} kg (Supply limit: {grade_supply[g]})')
    print('\n--- Brand Blending Proportions ---')
    for b in brands:
        prod = total_prod[b].getValue()
        if prod > 1e-06:
            print(f'Brand {b}:')
            for g in grades:
                prop = x_vars[g, b].X / prod if prod > 1e-08 else 0.0
                print(f'  Grade {g}: {prop * 100:.2f}%')
else:
    print(f'No optimal solution found. Status: {m.status}')