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
 'route': 'RA',
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
from gurobipy import GRB
import pandas as pd

def solve_problem():
    df_grades = CSVQA_FRAMES['file_0_view_0']
    df_brands = CSVQA_FRAMES['file_1_view_0']
    G = list(df_grades['Grade'])
    B = list(df_brands['Brand'])
    S_g = {}
    C_g = {}
    for (_, row) in df_grades.iterrows():
        g = row['Grade']
        S_g[g] = float(row['Daily Supply (kg)'])
        C_g[g] = float(row['Cost (CNY/kg)'])
    P_b = {}
    for (_, row) in df_brands.iterrows():
        b = row['Brand']
        P_b[b] = float(row['Selling Price (CNY/kg)'])
    L_gb = {}
    U_gb = {}
    for (_, row) in df_brands.iterrows():
        b = row['Brand']
        req = row['Blending Requirements']
        tokens = req.replace('  ', ' ').split(' ')
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in G:
                g = token
                if i + 2 < len(tokens):
                    if tokens[i + 1].casefold() == 'less' and tokens[i + 2].casefold().startswith('than'):
                        percent_str = tokens[i + 2][len('than'):] if tokens[i + 2].casefold().startswith('than') else tokens[i + 2]
                        percent_str = percent_str.strip('%')
                        if percent_str == '':
                            i += 3
                            percent_str = tokens[i] if i < len(tokens) else ''
                        ub = float(percent_str) / 100.0
                        U_gb[g, b] = ub
                        i += 3
                        continue
                    elif tokens[i + 1].casefold() == 'more' and tokens[i + 2].casefold().startswith('than'):
                        percent_str = tokens[i + 2][len('than'):] if tokens[i + 2].casefold().startswith('than') else tokens[i + 2]
                        percent_str = percent_str.strip('%')
                        if percent_str == '':
                            i += 3
                            percent_str = tokens[i] if i < len(tokens) else ''
                        lb = float(percent_str) / 100.0
                        L_gb[g, b] = lb
                        i += 3
                        continue
            i += 1
    m = gp.Model('wine_blending')
    x_keys = [(g, b) for g in G for b in B]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = {}
    for b in B:
        y_vars[b] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{b}')
        m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in G)), name=f'ydef_{b}')
    m.setObjective(gp.quicksum((P_b[b] * y_vars[b] for b in B)) - gp.quicksum((C_g[g] * gp.quicksum((x_vars[g, b] for b in B)) for g in G)), GRB.MAXIMIZE)
    for (g, b) in x_keys:
        if (g, b) in L_gb:
            m.addConstr(x_vars[g, b] >= L_gb[g, b] * y_vars[b], name=f'blend_lb_{g}_{b}')
        if (g, b) in U_gb:
            m.addConstr(x_vars[g, b] <= U_gb[g, b] * y_vars[b], name=f'blend_ub_{g}_{b}')
    for g in G:
        m.addConstr(gp.quicksum((x_vars[g, b] for b in B)) <= S_g[g], name=f'supply_{g}')
    if 'Red' in B:
        m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
    else:
        raise ValueError("Brand 'Red' not found in brands list.")
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')