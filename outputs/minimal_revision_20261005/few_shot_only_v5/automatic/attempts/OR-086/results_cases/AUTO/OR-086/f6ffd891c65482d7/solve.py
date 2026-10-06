import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    df_grades['Grade'] = df_grades['Grade'].astype(str).str.strip()
    grades = list(df_grades['Grade'])
    df_brands['Brand'] = df_brands['Brand'].astype(str).str.strip()
    brands = list(df_brands['Brand'])
    cost = {}
    supply = {}
    for (_, row) in df_grades.iterrows():
        g = str(row['Grade']).strip()
        if g in cost or g in supply:
            raise ValueError(f"Duplicate grade '{g}' in 30-1.csv")
        cost[g] = float(row['Cost (CNY/kg)'])
        supply[g] = float(row['Daily Supply (kg)'])
    price = {}
    for (_, row) in df_brands.iterrows():
        b = str(row['Brand']).strip()
        if b in price:
            raise ValueError(f"Duplicate brand '{b}' in 30-2.csv")
        price[b] = float(row['Selling Price (CNY/kg)'])
    blend_reqs = {}
    for (_, row) in df_brands.iterrows():
        b = str(row['Brand']).strip()
        reqs = []
        req_str = str(row['Blending Requirements'])
        for part in re.split(';|,| and ', req_str):
            part = part.strip()
            if not part:
                continue
            m = re.match('([A-Za-z0-9]+)\\s+(less than|more than|greater than|at least|at most|no more than|no less than|not less than|not more than|less|more|greater|<|>|≤|≥|=)\\s*([0-9.]+)%', part, re.IGNORECASE)
            if not m:
                raise ValueError(f"Cannot parse blending requirement: '{part}' for brand '{b}'")
            (g_raw, sense_raw, val_raw) = m.groups()
            g = g_raw.strip()
            sense = sense_raw.strip().lower()
            val = float(val_raw) / 100.0
            if sense in ['less than', 'less', '<']:
                sense = '<'
            elif sense in ['more than', 'greater than', 'greater', '>']:
                sense = '>'
            elif sense in ['at least', 'no less than', 'not less than', '≥']:
                sense = '>='
            elif sense in ['at most', 'no more than', 'not more than', '≤']:
                sense = '<='
            elif sense in ['=']:
                sense = '='
            else:
                raise ValueError(f"Unknown sense '{sense_raw}' in blending requirement '{part}'")
            reqs.append((g, sense, val))
        blend_reqs[b] = reqs
    for (b, reqs) in blend_reqs.items():
        for (g, _, _) in reqs:
            if g not in grades:
                raise ValueError(f"Grade '{g}' in blending requirements for brand '{b}' not found in 30-1.csv")
    m = gp.Model('WineBlending')
    m.setParam('MIPGap', 0.0001)
    keys = [(g, b) for g in grades for b in brands]
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        S_b = gp.quicksum((x[g, b] for g in grades))
        for (g, sense, val) in blend_reqs.get(b, []):
            if sense == '<':
                m.addConstr(x[g, b] <= val * S_b, name='')
            elif sense == '>':
                m.addConstr(x[g, b] >= val * S_b, name='')
            elif sense == '<=':
                m.addConstr(x[g, b] <= val * S_b, name='')
            elif sense == '>=':
                m.addConstr(x[g, b] >= val * S_b, name='')
            elif sense == '=':
                m.addConstr(x[g, b] == val * S_b, name='')
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirements for brand '{b}'")
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name='')
    red_brand = None
    for b in brands:
        if b.casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No brand named 'Red' found in 30-2.csv")
    m.addConstr(gp.quicksum((x[g, red_brand] for g in grades)) >= 2000, name='')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for g in grades:
            for b in brands:
                var = x[g, b]
                print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()