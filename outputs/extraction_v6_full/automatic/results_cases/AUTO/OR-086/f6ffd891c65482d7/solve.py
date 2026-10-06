import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(brands, blending_req_series, grades):
    blending_bounds = {brand: {g: (None, None) for g in grades} for brand in brands}
    less_pat = re.compile('([IV]+)\\s*less\\s*than\\s*([0-9]+)%', re.IGNORECASE)
    more_pat = re.compile('([IV]+)\\s*more\\s*than\\s*([0-9]+)%', re.IGNORECASE)
    for brand, req_str in zip(brands, blending_req_series):
        reqs = re.split('\\s{2,}', req_str.strip())
        for req in reqs:
            req = req.strip()
            m_less = less_pat.match(req)
            m_more = more_pat.match(req)
            if m_less:
                grade = m_less.group(1).strip()
                percent = float(m_less.group(2))
                prev = blending_bounds[brand][grade]
                blending_bounds[brand][grade] = (prev[0], percent / 100)
            elif m_more:
                grade = m_more.group(1).strip()
                percent = float(m_more.group(2))
                prev = blending_bounds[brand][grade]
                blending_bounds[brand][grade] = (percent / 100, prev[1])
            elif req == '':
                continue
            else:
                raise ValueError(f"Unrecognized blending requirement: '{req}' for brand '{brand}'")
    return blending_bounds

def solve_blending():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    grades = [g.strip() for g in df_grades['Grade']]
    brands = [b.strip() for b in df_brands['Brand']]
    supply = {row['Grade'].strip(): int(row['Daily Supply (kg)']) for _, row in df_grades.iterrows()}
    cost = {row['Grade'].strip(): float(row['Cost (CNY/kg)']) for _, row in df_grades.iterrows()}
    selling_price = {row['Brand'].strip(): float(row['Selling Price (CNY/kg)']) for _, row in df_brands.iterrows()}
    blending_bounds = parse_blending_requirements(brands, df_brands['Blending Requirements'], grades)
    m = gp.Model('WineBlending')
    x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='x')
    total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        total_prod = gp.quicksum((x[g, b] for g in grades))
        for g in grades:
            lb, ub = blending_bounds[b][g]
            if lb is not None:
                m.addConstr(x[g, b] >= lb * total_prod, name=f'blend_lb_{g}_{b}')
            if ub is not None:
                m.addConstr(x[g, b] <= ub * total_prod, name=f'blend_ub_{g}_{b}')
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    red_brand = None
    for b in brands:
        if b.strip().casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No 'Red' brand found in brands list.")
    m.addConstr(gp.quicksum((x[g, red_brand] for g in grades)) >= 2000, name='minprod_Red')
    m.optimize()
    return m
m = solve_blending()