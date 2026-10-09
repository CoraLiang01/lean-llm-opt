import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parse a blending requirements string into a list of (grade, sense, bound) tuples.
    E.g. "I less than 10%  II more than 50%" -> [("I", "<", 0.10), ("II", ">", 0.50)]
    """
    reqs = []
    pattern = '([IV]+)\\s+(less|more)\\s+than\\s+(\\d+)%'
    for match in re.finditer(pattern, req_str, flags=re.IGNORECASE):
        grade = match.group(1).strip()
        sense = '<' if match.group(2).casefold() == 'less' else '>'
        percent = float(match.group(3)) / 100.0
        reqs.append((grade, sense, percent))
    return reqs

def solve_problem():
    grades_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    brands_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    grades_df = pd.read_csv(grades_path, sep=',', dtype=str, keep_default_na=False)
    brands_df = pd.read_csv(brands_path, sep=',', dtype=str, keep_default_na=False)
    grades_df['Grade'] = grades_df['Grade'].str.strip()
    brands_df['Brand'] = brands_df['Brand'].str.strip()
    grades = list(grades_df['Grade'])
    brands = list(brands_df['Brand'])
    supply = {}
    cost = {}
    for (_, row) in grades_df.iterrows():
        g = row['Grade']
        try:
            supply[g] = float(row['Daily Supply (kg)'])
            cost[g] = float(row['Cost (CNY/kg)'])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
    price = {}
    blendreq = {}
    for (_, row) in brands_df.iterrows():
        b = row['Brand']
        try:
            price[b] = float(row['Selling Price (CNY/kg)'])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
        blendreq[b] = parse_blending_requirements(row['Blending Requirements'])
    for b in brands:
        for (g, _, _) in blendreq[b]:
            if g not in grades:
                raise ValueError(f"Blending requirement references unknown grade '{g}' for brand '{b}'.")
    m = gp.Model('WineBlending')
    m.setParam('MIPGap', 0.0001)
    keys = [(g, b) for g in grades for b in brands]
    x_vars = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * x_vars[g, b] for g in grades for b in brands))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        total_b = gp.quicksum((x_vars[g, b] for g in grades))
        for (g, sense, bound) in blendreq[b]:
            if sense == '<':
                m.addConstr(x_vars[g, b] <= bound * total_b, name='')
            elif sense == '>':
                m.addConstr(x_vars[g, b] >= bound * total_b, name='')
            else:
                raise ValueError(f"Unknown blending requirement sense '{sense}' for brand '{b}'.")
    for g in grades:
        m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply[g], name='')
    red_brand = None
    for b in brands:
        if b.casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No brand named 'Red' found in 30-2.csv.")
    m.addConstr(gp.quicksum((x_vars[g, red_brand] for g in grades)) >= 2000, name='')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')