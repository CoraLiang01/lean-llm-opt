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
 'route': 'NRM',
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
             'table_id': 'file_1_view_0'},
            {'columns': ['Brand'],
             'file_index': 1,
             'file_name': '30-2.csv',
             'filters': {'conditions': [{'column': 'Brand',
                                         'dtype': 'string',
                                         'evidence': 'the Red brand must have a daily production level of at least '
                                                     '2,000 kg.',
                                         'operator': 'exact',
                                         'value': 'Red'}],
                         'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Brand': 'Red'}}],
             'returned_rows': 1,
             'role': 'minimum production constraint',
             'table_id': 'file_1_view_1'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
import re
CSVQA_DATA = globals()['CSVQA_DATA']
table_raw = None
table_brand = None
table_red = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table_raw = t
    elif t['table_id'] == 'file_1_view_0':
        table_brand = t
    elif t['table_id'] == 'file_1_view_1':
        table_red = t
if table_raw is None or table_brand is None or table_red is None:
    raise RuntimeError('Missing required tables in CSVQA_DATA.')
G = [r['values']['Grade'] for r in table_raw['records']]
B = [r['values']['Brand'] for r in table_brand['records']]
S_g = {}
C_g = {}
for r in table_raw['records']:
    g = r['values']['Grade']
    try:
        S_g[g] = float(r['values']['Daily Supply (kg)'])
        C_g[g] = float(r['values']['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in raw table for grade {g}: {e}')
P_b = {}
for r in table_brand['records']:
    b = r['values']['Brand']
    try:
        P_b[b] = float(r['values']['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in brand table for brand {b}: {e}')
L_bg = {b: {g: 0.0 for g in G} for b in B}
U_bg = {b: {g: 1.0 for g in G} for b in B}
for r in table_brand['records']:
    b = r['values']['Brand']
    req = r['values']['Blending Requirements']
    clauses = re.findall('([A-Za-z0-9]+)\\s+(less than|more than)\\s+([0-9]+)%', req)
    if not req.strip() or (req.strip() and (not clauses)):
        if req.strip():
            raise ValueError(f"Unmatched blending requirement clause in brand '{b}': '{req}'")
        continue
    for grade, op, val in clauses:
        if grade not in G:
            raise ValueError(f"Unknown grade '{grade}' in blending requirements for brand '{b}'")
        percent = float(val) / 100.0
        if op == 'less than':
            U_bg[b][grade] = min(U_bg[b][grade], percent)
        elif op == 'more than':
            L_bg[b][grade] = max(L_bg[b][grade], percent)
        else:
            raise ValueError(f"Unknown blending operator '{op}' in brand '{b}'")
red_brand = None
for r in table_red['records']:
    if r['values']['Brand'] == 'Red':
        red_brand = 'Red'
if red_brand is None:
    raise ValueError('Red brand not found for minimum production constraint.')
m = gp.Model('wine_blending')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(B, G, lb=0.0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(B, lb=0.0, vtype=GRB.CONTINUOUS, name='')
for b in B:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[b, g] for g in G)), name=f'ydef_{b}')
for b in B:
    for g in G:
        m.addConstr(x_vars[b, g] >= L_bg[b][g] * y_vars[b], name=f'blend_lb_{b}_{g}')
        m.addConstr(x_vars[b, g] <= U_bg[b][g] * y_vars[b], name=f'blend_ub_{b}_{g}')
for g in G:
    m.addConstr(gp.quicksum((x_vars[b, g] for b in B)) <= S_g[g], name=f'supply_{g}')
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
m.setObjective(gp.quicksum((P_b[b] * y_vars[b] for b in B)) - gp.quicksum((C_g[g] * gp.quicksum((x_vars[b, g] for b in B)) for g in G)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')