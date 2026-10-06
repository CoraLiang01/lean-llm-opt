import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_blending_requirements(req_str):
    """
    Parses a blending requirements string like:
    "I less than 10%  II more than 50%"
    Returns a list of tuples: (grade, sense, value)
    where sense is 'lt' or 'gt', value is a float (proportion, e.g., 0.10)
    """
    reqs = []
    pattern = '([I]{1,3})\\s+(less than|more than)\\s+(\\d+)%'
    for match in re.finditer(pattern, req_str):
        grade = match.group(1).strip()
        sense = 'lt' if match.group(2).strip() == 'less than' else 'gt'
        value = float(match.group(3)) / 100.0
        reqs.append((grade, sense, value))
    return reqs

def solve_problem():
    path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df_grades = pd.read_csv(path1, sep=',')
    df_brands = pd.read_csv(path2, sep=',')
    grades = [g.strip() for g in df_grades['Grade']]
    grades_set = set(grades)
    brands = [b.strip() for b in df_brands['Brand']]
    brands_set = set(brands)
    supply = {}
    cost = {}
    for idx, row in df_grades.iterrows():
        g = row['Grade'].strip()
        supply[g] = float(row['Daily Supply (kg)'])
        cost[g] = float(row['Cost (CNY/kg)'])
    selling_price = {}
    for idx, row in df_brands.iterrows():
        b = row['Brand'].strip()
        selling_price[b] = float(row['Selling Price (CNY/kg)'])
    blending_reqs = {}
    for idx, row in df_brands.iterrows():
        b = row['Brand'].strip()
        req_str = str(row['Blending Requirements'])
        blending_reqs[b] = parse_blending_requirements(req_str)
    m = gp.Model('WineBlending')
    x = m.addVars(grades, brands, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_revenue = gp.quicksum((selling_price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
    total_cost = gp.quicksum((cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades))
    m.setObjective(total_revenue - total_cost, gp.GRB.MAXIMIZE)
    for b in brands:
        reqs = blending_reqs[b]
        total_prod = gp.quicksum((x[g, b] for g in grades))
        for g_req, sense, value in reqs:
            if g_req not in grades_set:
                raise ValueError(f"Grade '{g_req}' in blending requirements for brand '{b}' not found in grades list.")
            if sense == 'lt':
                m.addConstr(x[g_req, b] <= value * total_prod, name=f'blend_{b}_{g_req}_lt')
            elif sense == 'gt':
                m.addConstr(x[g_req, b] >= value * total_prod, name=f'blend_{b}_{g_req}_gt')
            else:
                raise ValueError(f"Unknown blending sense '{sense}' for brand '{b}'.")
    for g in grades:
        m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    red_brand = None
    for b in brands:
        if b.strip().casefold() == 'red':
            red_brand = b
            break
    if red_brand is None:
        raise ValueError("Brand 'Red' not found in brands list.")
    m.addConstr(gp.quicksum((x[g, red_brand] for g in grades)) >= 2000, name='minprod_Red')
    m.optimize()
    return m
m = solve_problem()