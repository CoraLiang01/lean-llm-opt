import pandas as pd
import numpy as np
import re
from gurobipy import Model, GRB, quicksum
grades_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', dtype=str, keep_default_na=False)
grades_df['Grade'] = grades_df['Grade'].str.strip()
grades_df['Daily Supply (kg)'] = grades_df['Daily Supply (kg)'].astype(float)
grades_df['Cost (CNY/kg)'] = grades_df['Cost (CNY/kg)'].astype(float)
brands_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', dtype=str, keep_default_na=False)
brands_df['Brand'] = brands_df['Brand'].str.strip()
brands_df['Selling Price (CNY/kg)'] = brands_df['Selling Price (CNY/kg)'].astype(float)
brands_df['Blending Requirements'] = brands_df['Blending Requirements'].str.strip()
grades = list(grades_df['Grade'])
brands = list(brands_df['Brand'])
grade_supply = dict(zip(grades_df['Grade'], grades_df['Daily Supply (kg)']))
grade_cost = dict(zip(grades_df['Grade'], grades_df['Cost (CNY/kg)']))
brand_price = dict(zip(brands_df['Brand'], brands_df['Selling Price (CNY/kg)']))
blending_bounds = {brand: {} for brand in brands}
for idx, row in brands_df.iterrows():
    brand = row['Brand']
    reqs = row['Blending Requirements']
    req_list = re.split('\\s{2,}', reqs)
    for req in req_list:
        req = req.strip()
        m = re.match('^([IVX]+)\\s+(less|more)\\s+than\\s+(\\d+)%$', req, re.IGNORECASE)
        if m:
            grade = m.group(1).strip()
            bound_type = m.group(2).strip().casefold()
            percent = float(m.group(3)) / 100.0
            if bound_type == 'less':
                blending_bounds[brand][grade] = ('upper', percent)
            elif bound_type == 'more':
                blending_bounds[brand][grade] = ('lower', percent)
            else:
                raise ValueError(f'Unknown bound type in blending requirement: {req}')
        elif req == '':
            continue
        else:
            raise ValueError(f"Unrecognized blending requirement format: '{req}' for brand '{brand}'")
for brand in brands:
    for grade in blending_bounds[brand]:
        if grade not in grades:
            raise ValueError(f"Grade '{grade}' in blending requirements for brand '{brand}' not found in grades list.")
m = Model('wine_blending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
total_sales = quicksum((quicksum((x_vars[grade, brand] for grade in grades)) * brand_price[brand] for brand in brands))
total_cost = quicksum((x_vars[grade, brand] * grade_cost[grade] for grade in grades for brand in brands))
m.setObjective(total_sales - total_cost, GRB.MAXIMIZE)
for brand in brands:
    total_brand = quicksum((x_vars[grade, brand] for grade in grades))
    for grade, (bound_type, value) in blending_bounds[brand].items():
        if bound_type == 'upper':
            m.addConstr(x_vars[grade, brand] <= value * total_brand, name='')
        elif bound_type == 'lower':
            m.addConstr(x_vars[grade, brand] >= value * total_brand, name='')
        else:
            raise ValueError(f"Unknown bound type '{bound_type}' for blending requirement.")
for grade in grades:
    m.addConstr(quicksum((x_vars[grade, brand] for brand in brands)) <= grade_supply[grade], name='')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(quicksum((x_vars[grade, 'Red'] for grade in grades)) >= 2000, name='')
m.optimize()