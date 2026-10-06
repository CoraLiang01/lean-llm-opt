import pandas as pd
import numpy as np
import re
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    df_grades['Grade'] = df_grades['Grade'].astype(str).str.strip()
    df_brands['Brand'] = df_brands['Brand'].astype(str).str.strip()
    grades = list(df_grades['Grade'])
    brands = list(df_brands['Brand'])
    Supply = dict(zip(df_grades['Grade'], df_grades['Daily Supply (kg)']))
    Cost = dict(zip(df_grades['Grade'], df_grades['Cost (CNY/kg)']))
    Price = dict(zip(df_brands['Brand'], df_brands['Selling Price (CNY/kg)']))
    blending_bounds = {b: {} for b in brands}
    for idx, row in df_brands.iterrows():
        b = row['Brand']
        reqs = row['Blending Requirements']
        req_list = re.split('\\s{2,}', reqs.strip())
        for req in req_list:
            req = req.strip()
            m_less = re.match('([A-Za-z0-9]+)\\s+less than\\s+([0-9]+)%', req, re.IGNORECASE)
            m_more = re.match('([A-Za-z0-9]+)\\s+more than\\s+([0-9]+)%', req, re.IGNORECASE)
            if m_less:
                g = m_less.group(1).strip()
                ub = float(m_less.group(2)) / 100.0
                if g not in blending_bounds[b]:
                    blending_bounds[b][g] = [None, None]
                blending_bounds[b][g][1] = ub
            elif m_more:
                g = m_more.group(1).strip()
                lb = float(m_more.group(2)) / 100.0
                if g not in blending_bounds[b]:
                    blending_bounds[b][g] = [None, None]
                blending_bounds[b][g][0] = lb
            else:
                raise ValueError(f"Unrecognized blending requirement: '{req}' for brand '{b}'")
    m = gp.Model('wine_blending')
    x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
    total_prod = {}
    for b in brands:
        total_prod[b] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'totprod_{b}')
    for b in brands:
        m.addConstr(total_prod[b] == gp.quicksum((x[g, b] for g in grades)), name=f'totalprod_link_{b}')
    for b in brands:
        for g in blending_bounds[b]:
            lb, ub = blending_bounds[b][g]
            if lb is not None:
                m.addConstr(x[g, b] >= lb * total_prod[b], name=f'blend_lb_{g}_{b}')
            if ub is not None:
                m.addConstr(x[g, b] <= ub * total_prod[b], name=f'blend_ub_{g}_{b}')
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= Supply[g], name=f'supply_{g}')
    if 'Red' not in brands:
        raise ValueError("Brand 'Red' not found in brands list.")
    m.addConstr(total_prod['Red'] >= 2000, name='minprod_Red')
    sales = gp.quicksum((total_prod[b] * Price[b] for b in brands))
    cost = gp.quicksum((x[g, b] * Cost[g] for g in grades for b in brands))
    m.setObjective(sales - cost, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()