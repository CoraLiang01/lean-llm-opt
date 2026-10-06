import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str, grades_set):
    """
    Parse a blending requirements string into a dict:
    {grade: {'lb': float or None, 'ub': float or None}}
    """
    reqs = {}
    req_str = re.sub('\\s+', ' ', req_str.strip())
    pattern = '([A-Za-z0-9]+)\\s+(less than|more than|at least|no more than|>=|<=|>|<|=)\\s*([0-9]+)%'
    for match in re.finditer(pattern, req_str, flags=re.IGNORECASE):
        (grade, op, percent) = match.groups()
        grade = grade.strip()
        percent = float(percent) / 100.0
        if grade not in grades_set:
            continue
        if grade not in reqs:
            reqs[grade] = {'lb': None, 'ub': None}
        op = op.casefold()
        if op in ['less than', '<', 'no more than', '<=']:
            reqs[grade]['ub'] = percent
        elif op in ['more than', '>', 'at least', '>=']:
            reqs[grade]['lb'] = percent
        elif op in ['=']:
            reqs[grade]['lb'] = percent
            reqs[grade]['ub'] = percent
    return reqs

def solve_blending():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    df_grades['Grade'] = df_grades['Grade'].astype(str).str.strip()
    df_brands['Brand'] = df_brands['Brand'].astype(str).str.strip()
    grades = list(df_grades['Grade'])
    brands = list(df_brands['Brand'])
    supply = dict(zip(df_grades['Grade'], df_grades['Daily Supply (kg)']))
    cost = dict(zip(df_grades['Grade'], df_grades['Cost (CNY/kg)']))
    price = dict(zip(df_brands['Brand'], df_brands['Selling Price (CNY/kg)']))
    blending_reqs = {}
    for (idx, row) in df_brands.iterrows():
        brand = row['Brand']
        req_str = row['Blending Requirements']
        blending_reqs[brand] = parse_blending_requirements(req_str, set(grades))
    keys = [(g, b) for g in grades for b in brands]
    m = gp.Model('wine_blending')
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        reqs = blending_reqs.get(b, {})
        total_prod = gp.quicksum((x[g, b] for g in grades))
        for g in grades:
            bounds = reqs.get(g, {})
            lb = bounds.get('lb', None)
            ub = bounds.get('ub', None)
            if ub is not None:
                m.addConstr(x[g, b] <= ub * total_prod, name=f'blend_ub_{g}_{b}')
            if lb is not None:
                m.addConstr(x[g, b] >= lb * total_prod, name=f'blend_lb_{g}_{b}')
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    red_brand = None
    for b in brands:
        if b.casefold().strip() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No brand named 'Red' found in brands list.")
    m.addConstr(gp.quicksum((x[g, red_brand] for g in grades)) >= 2000, name='minprod_red')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_blending()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')