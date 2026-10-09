import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parse a blending requirements string like:
        "I less than 10%  II more than 50%"
    into a list of tuples: (grade, sense, value)
    where sense is 'le' or 'ge', value is a float in [0,1]
    """
    reqs = []
    tokens = re.split('\\s{2,}', req_str.strip())
    for token in tokens:
        m = re.match('([IVX]+)\\s+(less|more)\\s+than\\s+(\\d+)%', token.strip(), re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            sense = m.group(2).strip().casefold()
            value = float(m.group(3)) / 100.0
            if sense == 'less':
                reqs.append((grade, 'le', value))
            elif sense == 'more':
                reqs.append((grade, 'ge', value))
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirement: '{token}'")
        else:
            raise ValueError(f"Could not parse blending requirement: '{token}'")
    return reqs

def solve_blending():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    df_grades['Grade'] = df_grades['Grade'].astype(str).str.strip()
    df_brands['Brand'] = df_brands['Brand'].astype(str).str.strip()
    df_brands['Blending Requirements'] = df_brands['Blending Requirements'].astype(str).str.strip()
    grades = list(df_grades['Grade'])
    brands = list(df_brands['Brand'])
    supply = dict(zip(df_grades['Grade'], df_grades['Daily Supply (kg)']))
    cost = dict(zip(df_grades['Grade'], df_grades['Cost (CNY/kg)']))
    selling_price = dict(zip(df_brands['Brand'], df_brands['Selling Price (CNY/kg)']))
    blending_reqs = {}
    for (idx, row) in df_brands.iterrows():
        brand = row['Brand']
        req_str = row['Blending Requirements']
        blending_reqs[brand] = parse_blending_requirements(req_str)
    for g in grades:
        if g not in supply or g not in cost:
            raise ValueError(f"Missing supply or cost for grade '{g}'")
    for b in brands:
        if b not in selling_price or b not in blending_reqs:
            raise ValueError(f"Missing selling price or blending requirements for brand '{b}'")
    keys = [(g, b) for g in grades for b in brands]
    m = gp.Model('WineBlending')
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        reqs = blending_reqs[b]
        total_prod = gp.quicksum((x[g, b] for g in grades))
        for (g_req, sense, value) in reqs:
            if g_req not in grades:
                raise ValueError(f"Grade '{g_req}' in blending requirements for brand '{b}' not found in grades list.")
            if sense == 'le':
                m.addConstr(x[g_req, b] <= value * total_prod, name=f'blend_{b}_{g_req}_le')
            elif sense == 'ge':
                m.addConstr(x[g_req, b] >= value * total_prod, name=f'blend_{b}_{g_req}_ge')
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirements.")
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    if 'Red' not in brands:
        raise ValueError("Brand 'Red' not found in brands list.")
    m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
    m.Params.MIPGap = 0.0001
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
m = solve_blending()