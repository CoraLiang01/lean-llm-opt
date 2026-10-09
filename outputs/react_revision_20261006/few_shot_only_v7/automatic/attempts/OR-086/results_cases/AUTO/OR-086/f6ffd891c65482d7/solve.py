import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    df_grades = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
    df_brands = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
    grades = df_grades['Grade'].tolist()
    brands = df_brands['Brand'].tolist()
    supply = {}
    cost = {}
    for (_, row) in df_grades.iterrows():
        g = row['Grade']
        try:
            supply[g] = float(row['Daily Supply (kg)'])
            cost[g] = float(row['Cost (CNY/kg)'])
        except Exception as e:
            raise ValueError(f"Invalid numeric value in 30-1.csv for grade '{g}': {e}")
    price = {}
    for (_, row) in df_brands.iterrows():
        b = row['Brand']
        try:
            price[b] = float(row['Selling Price (CNY/kg)'])
        except Exception as e:
            raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
    lower_bound = {(g, b): 0.0 for g in grades for b in brands}
    upper_bound = {(g, b): 1.0 for g in grades for b in brands}
    for (_, row) in df_brands.iterrows():
        b = row['Brand']
        reqs = row['Blending Requirements']
        if not isinstance(reqs, str) or not reqs.strip():
            continue
        for req in reqs.split(';'):
            req = req.strip()
            m1 = re.match('Proportion of\\s+([^\\s]+)\\s*<\\s*([\\d.]+)%', req, re.IGNORECASE)
            if m1:
                g = m1.group(1)
                val = float(m1.group(2)) / 100.0
                if g not in grades:
                    raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{b}'")
                upper_bound[g, b] = min(upper_bound[g, b], val)
                continue
            m2 = re.match('Proportion of\\s+([^\\s]+)\\s*>\\s*([\\d.]+)%', req, re.IGNORECASE)
            if m2:
                g = m2.group(1)
                val = float(m2.group(2)) / 100.0
                if g not in grades:
                    raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{b}'")
                lower_bound[g, b] = max(lower_bound[g, b], val)
                continue
            if req:
                raise ValueError(f"Unrecognized blending requirement: '{req}' for brand '{b}'")
    for g in grades:
        if g not in supply or g not in cost:
            raise ValueError(f"Missing supply or cost for grade '{g}'")
    for b in brands:
        if b not in price:
            raise ValueError(f"Missing selling price for brand '{b}'")
    for g in grades:
        for b in brands:
            if (g, b) not in lower_bound or (g, b) not in upper_bound:
                raise ValueError(f"Missing blending bounds for grade '{g}', brand '{b}'")
    m = gp.Model('WineBlending')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars([(g, b) for g in grades for b in brands], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_prod = {}
    for b in brands:
        total_prod[b] = gp.quicksum((x_vars[g, b] for g in grades))
    total_revenue = gp.quicksum((price[b] * total_prod[b] for b in brands))
    total_cost = gp.quicksum((cost[g] * x_vars[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        for g in grades:
            lb = lower_bound[g, b]
            ub = upper_bound[g, b]
            if lb > 0.0:
                m.addConstr(x_vars[g, b] >= lb * total_prod[b], name='')
            if ub < 1.0:
                m.addConstr(x_vars[g, b] <= ub * total_prod[b], name='')
    for g in grades:
        m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply[g], name='')
    red_brand = None
    for b in brands:
        if b.casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No brand named 'Red' found in 30-2.csv")
    m.addConstr(gp.quicksum((x_vars[g, red_brand] for g in grades)) >= 2000, name='')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')