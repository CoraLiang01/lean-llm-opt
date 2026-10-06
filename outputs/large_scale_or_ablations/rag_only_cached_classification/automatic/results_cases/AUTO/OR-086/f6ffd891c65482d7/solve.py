import pandas as pd
import numpy as np
import re
from gurobipy import Model, GRB
file_30_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
file_30_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
df_grades = pd.read_csv(file_30_1, sep=',')
df_brands = pd.read_csv(file_30_2, sep=',')
grades = df_grades['Grade'].astype(str).tolist()
brands = df_brands['Brand'].astype(str).tolist()
Supply = {row['Grade']: int(row['Daily Supply (kg)']) for _, row in df_grades.iterrows()}
Cost = {row['Grade']: float(row['Cost (CNY/kg)']) for _, row in df_grades.iterrows()}
Price = {row['Brand']: float(row['Selling Price (CNY/kg)']) for _, row in df_brands.iterrows()}

def parse_blend_req(req_str):
    """
    Parses a blending requirement string like:
    'I less than 10%  II more than 50%'
    Returns: dict of {grade: (lower, upper)}, where lower/upper are floats in [0,1] or None
    """
    bounds = {}
    pattern = '([A-Za-z0-9]+)\\s+(less|more)\\s+than\\s+([0-9]+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        sense = match.group(2).strip()
        percent = float(match.group(3))
        if sense == 'less':
            bounds.setdefault(grade, [None, None])
            bounds[grade][1] = percent / 100.0
        elif sense == 'more':
            bounds.setdefault(grade, [None, None])
            bounds[grade][0] = percent / 100.0
    for g in bounds:
        lb = bounds[g][0] if bounds[g][0] is not None else None
        ub = bounds[g][1] if bounds[g][1] is not None else None
        bounds[g] = (lb, ub)
    return bounds
BlendBounds = {}
for _, row in df_brands.iterrows():
    brand = str(row['Brand'])
    req_str = str(row['Blending Requirements'])
    bounds = parse_blend_req(req_str)
    full_bounds = {}
    for g in grades:
        if g in bounds:
            full_bounds[g] = bounds[g]
        else:
            full_bounds[g] = (None, None)
    BlendBounds[brand] = full_bounds
m = Model('wine_blending')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    total_b = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'total_{b}')
    m.addConstr(total_b == sum((x[g, b] for g in grades)), name=f'totaldef_{b}')
    for g in grades:
        lb, ub = BlendBounds[b][g]
        if lb is not None:
            m.addConstr(x[g, b] >= lb * total_b, name=f'blend_lb_{g}_{b}')
        if ub is not None:
            m.addConstr(x[g, b] <= ub * total_b, name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(sum((x[g, b] for b in brands)) <= Supply[g], name=f'supply_{g}')
m.addConstr(sum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
revenue = sum((sum((x[g, b] for g in grades)) * Price[b] for b in brands))
cost = sum((x[g, b] * Cost[g] for g in grades for b in brands))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
m.optimize()