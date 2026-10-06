import gurobipy as gp
import pandas as pd
import numpy as np
import re
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',')
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',')
grades_df['Grade'] = grades_df['Grade'].astype(str).str.strip()
brands_df['Brand'] = brands_df['Brand'].astype(str).str.strip()
grades = list(grades_df['Grade'])
brands = list(brands_df['Brand'])
supply = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
selling_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))

def parse_blending_req(req_str):
    reqs = {}
    tokens = re.split('\\s{2,}', req_str.strip())
    for token in tokens:
        token = token.strip()
        m = re.match('^([IVX]+)\\s+(less|more)\\s+than\\s+([0-9]+)%$', token, re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            bound_type = '<' if m.group(2).casefold() == 'less' else '>'
            value = float(m.group(3)) / 100.0
            reqs[grade] = (bound_type, value)
    return reqs
blending_req = {}
for (idx, row) in brands_df.iterrows():
    brand = row['Brand']
    req_str = row['Blending Requirements']
    blending_req[brand] = parse_blending_req(req_str)
keys = [(g, b) for g in grades for b in brands]

def solve_problem():
    m = gp.Model('WineBlending')
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_prod = {b: gp.quicksum((x[g, b] for g in grades)) for b in brands}
    total_sales = gp.quicksum((selling_price[b] * total_prod[b] for b in brands))
    total_cost = gp.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(total_sales - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        prod = total_prod[b]
        for g in grades:
            if g in blending_req[b]:
                (bound_type, value) = blending_req[b][g]
                if bound_type == '<':
                    m.addConstr(x[g, b] <= value * prod, name=f'blend_ub_{g}_{b}')
                elif bound_type == '>':
                    m.addConstr(x[g, b] >= value * prod, name=f'blend_lb_{g}_{b}')
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    if 'Red' in brands:
        m.addConstr(total_prod['Red'] >= 2000, name='minprod_Red')
    else:
        raise ValueError("Brand 'Red' not found in brands list.")
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')