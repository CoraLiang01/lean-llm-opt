import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str, grades_set):
    """
    Parse a blending requirement string into a list of (grade, sense, percent) tuples.
    E.g., "I less than 60%; II more than 20%" -> [("I", "<=", 60), ("II", ">=", 20)]
    """
    reqs = []
    if not isinstance(req_str, str) or not req_str.strip():
        return reqs
    for part in re.split('[;,]', req_str):
        part = part.strip()
        if not part:
            continue
        m = re.match('^([A-Za-z0-9]+)\\s+(less|more)\\s+than\\s+([0-9]+)%$', part, re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            sense = m.group(2).strip().lower()
            percent = float(m.group(3))
            if grade not in grades_set:
                raise ValueError(f"Blending requirement references unknown grade '{grade}'")
            if sense == 'less':
                reqs.append((grade, '<=', percent))
            elif sense == 'more':
                reqs.append((grade, '>=', percent))
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirement")
        else:
            m2 = re.match('^([A-Za-z0-9]+)\\s*(<=|>=|<|>)\\s*([0-9]+)%$', part)
            if m2:
                grade = m2.group(1).strip()
                op = m2.group(2)
                percent = float(m2.group(3))
                if grade not in grades_set:
                    raise ValueError(f"Blending requirement references unknown grade '{grade}'")
                if op == '<=' or op == '<':
                    reqs.append((grade, '<=', percent))
                elif op == '>=' or op == '>':
                    reqs.append((grade, '>=', percent))
                else:
                    raise ValueError(f"Unknown operator '{op}' in blending requirement")
            else:
                raise ValueError(f"Could not parse blending requirement part: '{part}'")
    return reqs

def solve_problem():
    grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', sep=',', dtype=str, keep_default_na=False)
    brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', sep=',', dtype=str, keep_default_na=False)
    grades = list(grades_df['Grade'].unique())
    brands = list(brands_df['Brand'].unique())
    grades_set = set(grades)
    brands_set = set(brands)
    supply = {}
    cost = {}
    for (_, row) in grades_df.iterrows():
        g = row['Grade']
        if g not in grades_set:
            continue
        try:
            supply[g] = float(row['Daily Supply (kg)'])
            cost[g] = float(row['Cost (CNY/kg)'])
        except Exception as e:
            raise ValueError(f"Invalid numeric value in 30-1.csv for grade '{g}': {e}")
    selling_price = {}
    blending_reqs = {}
    for (_, row) in brands_df.iterrows():
        b = row['Brand']
        if b not in brands_set:
            continue
        try:
            selling_price[b] = float(row['Selling Price (CNY/kg)'])
        except Exception as e:
            raise ValueError(f"Invalid numeric value in 30-2.csv for brand '{b}': {e}")
        req_str = row['Blending Requirements']
        blending_reqs[b] = parse_blending_requirements(req_str, grades_set)
    if set(supply.keys()) != grades_set or set(cost.keys()) != grades_set:
        raise ValueError('Mismatch in grades between supply/cost and grade set.')
    if set(selling_price.keys()) != brands_set:
        raise ValueError('Mismatch in brands between selling price and brand set.')
    x_keys = [(g, b) for g in grades for b in brands]
    m = gp.Model('WineBlending')
    x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for g in grades:
        m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    for b in brands:
        total_prod_b = gp.quicksum((x_vars[g, b] for g in grades))
        for (g_req, sense, percent) in blending_reqs[b]:
            if sense == '<=':
                m.addConstr(x_vars[g_req, b] <= percent / 100.0 * total_prod_b, name=f'blend_{b}_{g_req}_le')
            elif sense == '>=':
                m.addConstr(x_vars[g_req, b] >= percent / 100.0 * total_prod_b, name=f'blend_{b}_{g_req}_ge')
            else:
                raise ValueError(f"Unknown sense '{sense}' in blending requirements.")
    red_brand = None
    for b in brands:
        if b.casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("No brand named 'Red' found in 30-2.csv.")
    m.addConstr(gp.quicksum((x_vars[g, red_brand] for g in grades)) >= 2000, name='minprod_Red')
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