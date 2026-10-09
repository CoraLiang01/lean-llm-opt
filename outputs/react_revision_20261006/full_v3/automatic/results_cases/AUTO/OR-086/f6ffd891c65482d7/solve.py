import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(brands, grades, blend_req_series):
    blend_bounds = {(b, g): [0.0, 1.0] for b in brands for g in grades}
    for (b, req) in blend_req_series.items():
        tokens = re.split('\\s+', req.strip())
        i = 0
        while i < len(tokens):
            for g in grades:
                if tokens[i].casefold() == g.casefold():
                    if i + 2 < len(tokens):
                        if tokens[i + 1].casefold() == 'less' and tokens[i + 2].casefold() == 'than':
                            if i + 3 < len(tokens):
                                m = re.match('(\\d+(\\.\\d+)?)%', tokens[i + 3])
                                if m:
                                    val = float(m.group(1)) / 100.0
                                    blend_bounds[b, g][1] = min(blend_bounds[b, g][1], val)
                                    i += 4
                                    continue
                        elif tokens[i + 1].casefold() == 'more' and tokens[i + 2].casefold() == 'than':
                            if i + 3 < len(tokens):
                                m = re.match('(\\d+(\\.\\d+)?)%', tokens[i + 3])
                                if m:
                                    val = float(m.group(1)) / 100.0
                                    blend_bounds[b, g][0] = max(blend_bounds[b, g][0], val)
                                    i += 4
                                    continue
            i += 1
    return {(b, g): (blend_bounds[b, g][0], blend_bounds[b, g][1]) for b in brands for g in grades}

def solve_problem():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    grades = [g.strip() for g in df_grades['Grade']]
    brands = [b.strip() for b in df_brands['Brand']]
    supply = {}
    cost = {}
    for (_, row) in df_grades.iterrows():
        g = row['Grade'].strip()
        supply[g] = float(row['Daily Supply (kg)'])
        cost[g] = float(row['Cost (CNY/kg)'])
    selling_price = {}
    for (_, row) in df_brands.iterrows():
        b = row['Brand'].strip()
        selling_price[b] = float(row['Selling Price (CNY/kg)'])
    blend_req_series = df_brands.set_index(df_brands['Brand'].str.strip())['Blending Requirements']
    blend_bounds = parse_blending_requirements(brands, grades, blend_req_series)
    if set(grades) != set(supply.keys()) or set(grades) != set(cost.keys()):
        raise ValueError('Mismatch in grade identifiers between supply/cost and grade list.')
    if set(brands) != set(selling_price.keys()):
        raise ValueError('Mismatch in brand identifiers between selling price and brand list.')
    for b in brands:
        for g in grades:
            if (b, g) not in blend_bounds:
                raise ValueError(f'Missing blending bounds for brand {b}, grade {g}.')
    m = gp.Model('WineBlending')
    m.Params.MIPGap = 0.0001
    keys = [(g, b) for g in grades for b in brands]
    x = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        total_b = gp.quicksum((x[g, b] for g in grades))
        for g in grades:
            (lb, ub) = blend_bounds[b, g]
            if lb > 0.0:
                m.addConstr(x[g, b] >= lb * total_b, name='')
            if ub < 1.0:
                m.addConstr(x[g, b] <= ub * total_b, name='')
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name='')
    red_brand = None
    for b in brands:
        if b.casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No 'Red' brand found in brands list.")
    m.addConstr(gp.quicksum((x[g, red_brand] for g in grades)) >= 2000, name='')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for g in grades:
            for b in brands:
                var = x[g, b]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()