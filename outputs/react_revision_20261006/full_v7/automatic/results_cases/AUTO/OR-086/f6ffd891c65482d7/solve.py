import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parse a blending requirements string into a list of (grade, sense, percent) tuples.
    E.g., "I less than 10%  II more than 50%" -> [("I", "<", 10), ("II", ">", 50)]
    """
    reqs = []
    tokens = re.split('\\s{2,}', req_str.strip())
    for token in tokens:
        m = re.match('([IV]+)\\s+(less|more)\\s+than\\s+(\\d+)%', token.strip(), re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            sense = '<' if m.group(2).casefold() == 'less' else '>'
            percent = float(m.group(3))
            reqs.append((grade, sense, percent))
    return reqs

def solve_problem():
    grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
    brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
    grades_df['Grade'] = grades_df['Grade'].str.strip()
    brands_df['Brand'] = brands_df['Brand'].str.strip()
    grades = list(grades_df['Grade'])
    brands = list(brands_df['Brand'])
    supply_limit = {}
    cost = {}
    for (_, row) in grades_df.iterrows():
        g = row['Grade']
        try:
            supply_limit[g] = float(row['Daily Supply (kg)'])
            cost[g] = float(row['Cost (CNY/kg)'])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
    selling_price = {}
    blending_reqs = {}
    for (_, row) in brands_df.iterrows():
        b = row['Brand']
        try:
            selling_price[b] = float(row['Selling Price (CNY/kg)'])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
        blending_reqs[b] = parse_blending_requirements(row['Blending Requirements'])
    for g in grades:
        if g not in supply_limit or g not in cost:
            raise ValueError(f'Missing supply/cost data for grade {g}')
    for b in brands:
        if b not in selling_price or b not in blending_reqs:
            raise ValueError(f'Missing selling price or blending requirements for brand {b}')
    keys = [(g, b) for g in grades for b in brands]
    m = gp.Model('WineBlending')
    x_vars = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x_vars[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for g in grades:
        m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_limit[g], name=f'supply_{g}')
    for b in brands:
        total_prod = gp.quicksum((x_vars[g, b] for g in grades))
        for (g_req, sense, percent) in blending_reqs[b]:
            if g_req not in grades:
                raise ValueError(f"Blending requirement references unknown grade '{g_req}' in brand '{b}'")
            if sense == '<':
                m.addConstr(x_vars[g_req, b] <= percent / 100.0 * total_prod, name=f'blend_ub_{g_req}_{b}')
            elif sense == '>':
                m.addConstr(x_vars[g_req, b] >= percent / 100.0 * total_prod, name=f'blend_lb_{g_req}_{b}')
            else:
                raise ValueError(f"Unknown blending sense '{sense}' in blending requirements for brand '{b}'")
    if 'Red' not in brands:
        raise ValueError("Brand 'Red' not found in brands list")
    m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000.0, name='minprod_Red')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')