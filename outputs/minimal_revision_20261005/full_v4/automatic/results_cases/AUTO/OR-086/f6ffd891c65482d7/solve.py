import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parses a blending requirements string like:
        'I less than 10%  II more than 50%'
    Returns a list of tuples: (grade, sense, value)
    where sense is 'le' or 'ge', and value is a float (proportion).
    """
    reqs = []
    pattern = '([IV]+)\\s+(less|more)\\s+than\\s+(\\d+)%'
    for match in re.finditer(pattern, req_str, flags=re.IGNORECASE):
        grade = match.group(1).strip()
        sense = match.group(2).strip().casefold()
        value = float(match.group(3)) / 100.0
        if sense == 'less':
            reqs.append((grade, 'le', value))
        elif sense == 'more':
            reqs.append((grade, 'ge', value))
        else:
            raise ValueError(f"Unknown sense '{sense}' in blending requirement: {match.group(0)}")
    return reqs

def solve_blending():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    df_grades['Grade_norm'] = df_grades['Grade'].astype(str).str.strip()
    df_brands['Brand_norm'] = df_brands['Brand'].astype(str).str.strip()
    grades = list(df_grades['Grade_norm'])
    brands = list(df_brands['Brand_norm'])
    if len(set(grades)) != len(grades):
        raise ValueError('Duplicate grade identifiers found in 30-1.csv')
    if len(set(brands)) != len(brands):
        raise ValueError('Duplicate brand identifiers found in 30-2.csv')
    supply = dict(zip(df_grades['Grade_norm'], df_grades['Daily Supply (kg)']))
    cost = dict(zip(df_grades['Grade_norm'], df_grades['Cost (CNY/kg)']))
    price = dict(zip(df_brands['Brand_norm'], df_brands['Selling Price (CNY/kg)']))
    blending_reqs = {}
    for (idx, row) in df_brands.iterrows():
        brand = row['Brand_norm']
        req_str = row['Blending Requirements']
        reqs = parse_blending_requirements(req_str)
        for (g, _, _) in reqs:
            if g not in grades:
                raise ValueError(f"Grade '{g}' in blending requirements for brand '{brand}' not found in grades list.")
        blending_reqs[brand] = reqs
    keys = [(b, g) for b in brands for g in grades]
    m = gp.Model('wine_blending')
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((price[b] * gp.quicksum((x[b, g] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x[b, g] for b in brands for g in grades))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        reqs = blending_reqs.get(b, [])
        total_b = gp.quicksum((x[b, g] for g in grades))
        for (g, sense, val) in reqs:
            if sense == 'le':
                m.addConstr(x[b, g] - val * total_b <= 0, name=f'blend_{b}_{g}_le')
            elif sense == 'ge':
                m.addConstr(x[b, g] - val * total_b >= 0, name=f'blend_{b}_{g}_ge')
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirements.")
    for g in grades:
        m.addConstr(gp.quicksum((x[b, g] for b in brands)) <= supply[g], name=f'supply_{g}')
    red_brand = None
    for b in brands:
        if b.casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("Brand 'Red' not found in 30-2.csv")
    m.addConstr(gp.quicksum((x[red_brand, g] for g in grades)) >= 2000, name='minprod_red')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for b in brands:
            for g in grades:
                var = x[b, g]
                print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_blending()