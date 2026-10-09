import pandas as pd
import numpy as np
import re
from gurobipy import Model, GRB
grades_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
brands_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
grades_df = pd.read_csv(grades_path, sep=',')
grades_df['Grade'] = grades_df['Grade'].astype(str).str.strip()
grades = list(grades_df['Grade'])
supply_limit = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
brands_df = pd.read_csv(brands_path, sep=',')
brands_df['Brand'] = brands_df['Brand'].astype(str).str.strip()
brands = list(brands_df['Brand'])
price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))
blending_bounds = {b: {g: {'lb': None, 'ub': None} for g in grades} for b in brands}
less_than_pat = re.compile('(\\w+)\\s+less\\s+than\\s+(\\d+)%', re.IGNORECASE)
more_than_pat = re.compile('(\\w+)\\s+more\\s+than\\s+(\\d+)%', re.IGNORECASE)
for (idx, row) in brands_df.iterrows():
    brand = row['Brand']
    reqs = row['Blending Requirements']
    reqs_split = re.split('\\s{2,}', reqs.strip())
    for req in reqs_split:
        req = req.strip()
        m = less_than_pat.match(req)
        if m:
            grade = m.group(1).strip()
            percent = float(m.group(2))
            if grade not in grades:
                raise ValueError(f"Unknown grade '{grade}' in blending requirements for brand '{brand}'")
            blending_bounds[brand][grade]['ub'] = percent / 100.0
            continue
        m = more_than_pat.match(req)
        if m:
            grade = m.group(1).strip()
            percent = float(m.group(2))
            if grade not in grades:
                raise ValueError(f"Unknown grade '{grade}' in blending requirements for brand '{brand}'")
            blending_bounds[brand][grade]['lb'] = percent / 100.0
            continue
        if req != '':
            raise ValueError(f"Unrecognized blending requirement: '{req}' for brand '{brand}'")
m = Model('wine_blending')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    total_prod = x.sum('*', b)
    for g in grades:
        bounds = blending_bounds[b][g]
        if bounds['lb'] is not None:
            m.addConstr(x[g, b] >= bounds['lb'] * total_prod)
        if bounds['ub'] is not None:
            m.addConstr(x[g, b] <= bounds['ub'] * total_prod)
for g in grades:
    m.addConstr(x.sum(g, '*') <= supply_limit[g])
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list")
m.addConstr(x.sum('*', 'Red') >= 2000)
sales_revenue = sum((x.sum('*', b) * price[b] for b in brands))
raw_material_cost = sum((x[g, b] * cost[g] for g in grades for b in brands))
net_profit = sales_revenue - raw_material_cost
m.setObjective(net_profit, GRB.MAXIMIZE)
m.optimize()