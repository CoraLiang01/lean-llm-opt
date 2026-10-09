CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company '
          'that maximizes total net profit. The company produces three distinct wine brands by blending three grades '
          'of raw wine material.\n'
          '\n'
          'The necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs '
          'for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all '
          'three brands.\n'
          '\n'
          'The model should incorporate the following constraints. First, the Blending Requirements constraints '
          'specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. '
          'Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all '
          'brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint '
          'requires that the Red brand must have a daily production level of at least 2,000 kg.\n'
          '\n'
          'The objective is to determine the quantity of each raw grade allocated to each brand, defined as the '
          'decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw '
          'material cost.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'],
             'file_index': 0,
             'file_name': '30-1.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}},
                         {'source_row': 1,
                          'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}},
                         {'source_row': 2,
                          'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}],
             'returned_rows': 3,
             'role': 'raw material supply and cost',
             'table_id': 'file_0_view_0'},
            {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'],
             'file_index': 1,
             'file_name': '30-2.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Blending Requirements': 'I less than 10%  II more than 50%',
                                     'Brand': 'Red',
                                     'Selling Price (CNY/kg)': '5.5'}},
                         {'source_row': 1,
                          'values': {'Blending Requirements': 'III less than 70%  I more than 20%',
                                     'Brand': 'Yellow',
                                     'Selling Price (CNY/kg)': '5'}},
                         {'source_row': 2,
                          'values': {'Blending Requirements': 'I less than 50%  II more than 10%',
                                     'Brand': 'Blue',
                                     'Selling Price (CNY/kg)': '4.8'}}],
             'returned_rows': 3,
             'role': 'brand blending requirements and selling price',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np
import sys
import re

def solve_problem(CSVQA_FRAMES):
    frame_grades = CSVQA_FRAMES['file_0_view_0']
    frame_brands = CSVQA_FRAMES['file_1_view_0']
    G = []
    S_g = {}
    C_g = {}
    for (source_row, row) in frame_grades.iterrows():
        grade = row['Grade']
        G.append(grade)
        try:
            S_g[grade] = float(row['Daily Supply (kg)'])
        except Exception:
            raise ValueError(f'Invalid Daily Supply (kg) for grade {grade}')
        try:
            C_g[grade] = float(row['Cost (CNY/kg)'])
        except Exception:
            raise ValueError(f'Invalid Cost (CNY/kg) for grade {grade}')
    B = []
    P_b = {}
    blending_reqs = {}
    for (source_row, row) in frame_brands.iterrows():
        brand = row['Brand']
        B.append(brand)
        try:
            P_b[brand] = float(row['Selling Price (CNY/kg)'])
        except Exception:
            raise ValueError(f'Invalid Selling Price (CNY/kg) for brand {brand}')
        blending_reqs[brand] = row['Blending Requirements']
    blend_constraints = {b: [] for b in B}
    for b in B:
        reqs = blending_reqs[b]
        reqs = reqs.replace('%', ' %')
        tokens = reqs.split()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in G:
                grade = token
                if i + 2 < len(tokens):
                    sense_token = tokens[i + 1].casefold()
                    value_token = tokens[i + 2]
                    if sense_token == 'less':
                        sense = '<'
                    elif sense_token == 'more':
                        sense = '>'
                    else:
                        raise ValueError(f"Unknown blending sense '{sense_token}' in '{reqs}'")
                    if value_token.endswith('%'):
                        value = float(value_token[:-1]) / 100.0
                    else:
                        value = float(value_token)
                        if value > 1.0:
                            value = value / 100.0
                    blend_constraints[b].append((grade, sense, value))
                    i += 3
                else:
                    raise ValueError(f"Malformed blending requirement for brand {b}: '{reqs}'")
            else:
                i += 1
    m = gp.Model('WineBlending')
    x_keys = []
    for b in B:
        for g in G:
            x_keys.append((b, g))
    x_vars = m.addVars(x_keys, lb=0.0, name='')
    y_vars = m.addVars(B, lb=0.0, name='')
    for b in B:
        m.addConstr(y_vars[b] == gp.quicksum((x_vars[b, g] for g in G)), name=f'y_def_{b}')
    revenue = gp.quicksum((P_b[b] * y_vars[b] for b in B))
    cost = gp.quicksum((C_g[g] * gp.quicksum((x_vars[b, g] for b in B)) for g in G))
    m.setObjective(revenue - cost, gp.GRB.MAXIMIZE)
    for g in G:
        m.addConstr(gp.quicksum((x_vars[b, g] for b in B)) <= S_g[g], name=f'supply_{g}')
    eps = 1e-06
    for b in B:
        for (g, sense, value) in blend_constraints[b]:
            if sense == '<':
                m.addConstr(x_vars[b, g] <= (value - eps) * y_vars[b], name=f'blend_{b}_{g}_ub')
            elif sense == '>':
                m.addConstr(x_vars[b, g] >= (value + eps) * y_vars[b], name=f'blend_{b}_{g}_lb')
            else:
                raise ValueError(f'Unknown sense {sense} in blending constraints.')
    if 'Red' in B:
        m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)