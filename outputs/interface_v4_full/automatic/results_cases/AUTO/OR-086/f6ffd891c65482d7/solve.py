import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parse a blending requirements string like:
      'I less than 10%  II more than 50%'
    into a dict: {grade: (lower, upper)}, where lower/upper are floats in [0,1] or None.
    """
    reqs = {}
    pattern = '([IV]+)\\s+(less|more)\\s+than\\s+(\\d+)%'
    for match in re.finditer(pattern, req_str):
        grade, sense, pct = match.groups()
        pct = float(pct) / 100.0
        if grade not in reqs:
            reqs[grade] = [None, None]
        if sense == 'less':
            reqs[grade][1] = pct
        elif sense == 'more':
            reqs[grade][0] = pct
    return {g: (l, u) for g, (l, u) in reqs.items()}

def solve_problem():
    path_30_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path_30_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path_30_1, sep=',')
    df_brands = pd.read_csv(path_30_2, sep=',')
    grades = list(df_grades['Grade'].astype(str))
    brands = list(df_brands['Brand'].astype(str))
    supply = df_grades.set_index('Grade')['Daily Supply (kg)'].astype(float).to_dict()
    cost = df_grades.set_index('Grade')['Cost (CNY/kg)'].astype(float).to_dict()
    price = df_brands.set_index('Brand')['Selling Price (CNY/kg)'].astype(float).to_dict()
    blend_req = {}
    for idx, row in df_brands.iterrows():
        brand = str(row['Brand'])
        req_str = str(row['Blending Requirements'])
        blend_req[brand] = parse_blending_requirements(req_str)
    m = gp.Model('WineBlending')
    x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        total_b = gp.quicksum((x[g, b] for g in grades))
        reqs = blend_req.get(b, {})
        for g in grades:
            if g in reqs:
                lower, upper = reqs[g]
                if lower is not None:
                    m.addConstr(x[g, b] >= lower * total_b, name=f'blend_lb_{g}_{b}')
                if upper is not None:
                    m.addConstr(x[g, b] <= upper * total_b, name=f'blend_ub_{g}_{b}')
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
    m.optimize()
    return m
m = solve_problem()