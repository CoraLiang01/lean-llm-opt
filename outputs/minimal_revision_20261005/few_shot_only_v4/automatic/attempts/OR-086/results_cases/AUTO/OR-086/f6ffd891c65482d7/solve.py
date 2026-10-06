import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_blending():
    path_30_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path_30_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path_30_1, sep=',')
    df_brands = pd.read_csv(path_30_2, sep=',')
    grades = list(df_grades['Grade'].astype(str))
    brands = list(df_brands['Brand'].astype(str))
    if set(grades) != set(['I', 'II', 'III']):
        raise ValueError(f'Grades in CSV do not match required set: {grades}')
    if set(brands) != set(['Red', 'Yellow', 'Blue']):
        raise ValueError(f'Brands in CSV do not match required set: {brands}')
    supply = df_grades.set_index('Grade')['Daily Supply (kg)'].astype(float).to_dict()
    cost = df_grades.set_index('Grade')['Cost (CNY/kg)'].astype(float).to_dict()
    price = df_brands.set_index('Brand')['Selling Price (CNY/kg)'].astype(float).to_dict()
    blend_reqs = {b: [] for b in brands}
    for b in brands:
        req_str = df_brands.loc[df_brands['Brand'].astype(str) == b, 'Blending Requirements'].values[0]
        reqs = re.findall('Grade\\s+([IVX]+)\\s*([<>]=?)\\s*([\\d.]+)%', req_str)
        for (g, op, val) in reqs:
            g = g.strip()
            if g not in grades:
                raise ValueError(f"Unknown grade '{g}' in blending requirements for brand '{b}'")
            val = float(val) / 100.0
            if op == '<':
                blend_reqs[b].append((g, 'le', val))
            elif op == '<=':
                blend_reqs[b].append((g, 'le', val))
            elif op == '>':
                blend_reqs[b].append((g, 'ge', val))
            elif op == '>=':
                blend_reqs[b].append((g, 'ge', val))
            else:
                raise ValueError(f"Unknown operator '{op}' in blending requirements for brand '{b}'")
    m = gp.Model('wine_blending')
    m.Params.MIPGap = 0.0001
    x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        total_b = gp.quicksum((x[g, b] for g in grades))
        for (g, sense, bound) in blend_reqs[b]:
            if sense == 'le':
                m.addConstr(x[g, b] - bound * total_b <= 0, name=f'blend_{g}_{b}_le')
            elif sense == 'ge':
                m.addConstr(x[g, b] - bound * total_b >= 0, name=f'blend_{g}_{b}_ge')
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirements.")
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for g in grades:
            for b in brands:
                print(f'{x[g, b].VarName} {x[g, b].X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_blending()