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
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
import re
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_index': 0, 'file_name': '30-1.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 3, 'records': [{'source_row': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'source_row': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'source_row': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'returned_rows': 3, 'role': 'raw material supply and cost', 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_index': 1, 'file_name': '30-2.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 3, 'records': [{'source_row': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'source_row': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'source_row': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'returned_rows': 3, 'role': 'brand blending requirements and selling price', 'table_id': 'file_1_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
raw_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        raw_table = t
        break
if raw_table is None:
    raise ValueError('Raw material table not found.')
G = []
S_g = {}
C_g = {}
for rec in raw_table['records']:
    g = rec['values']['Grade']
    G.append(g)
    try:
        S_g[g] = float(rec['values']['Daily Supply (kg)'])
        C_g[g] = float(rec['values']['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid data in raw material table for grade {g}: {e}')
brand_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_1_view_0':
        brand_table = t
        break
if brand_table is None:
    raise ValueError('Brand table not found.')
B = []
P_b = {}
L_bg = {}
U_bg = {}
for rec in brand_table['records']:
    b = rec['values']['Brand']
    B.append(b)
    try:
        P_b[b] = float(rec['values']['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid selling price for brand {b}: {e}')
    blend_str = rec['values']['Blending Requirements']
    reqs = re.findall('([I]{1,3})\\s+(less than|more than)\\s+(\\d+)%', blend_str)
    if not reqs and blend_str.strip():
        raise ValueError(f"Could not parse blending requirements for brand {b}: '{blend_str}'")
    for grade, sense, percent in reqs:
        if grade not in G:
            raise ValueError(f"Unknown grade '{grade}' in blending requirements for brand {b}")
        val = float(percent) / 100.0
        if sense == 'less than':
            U_bg[b, grade] = val
        elif sense == 'more than':
            L_bg[b, grade] = val
        else:
            raise ValueError(f"Unknown sense '{sense}' in blending requirements for brand {b}")
for g in G:
    if g not in S_g or g not in C_g:
        raise ValueError(f'Missing supply or cost for grade {g}')
for b in B:
    if b not in P_b:
        raise ValueError(f'Missing selling price for brand {b}')
m = gp.Model('WineBlending')
m.Params.MIPGap = 0.0001
x = m.addVars(B, G, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(B, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in B:
    m.addConstr(y[b] == gp.quicksum((x[b, g] for g in G)), name=f'ydef_{b}')
for (b, g), lb in L_bg.items():
    m.addConstr(x[b, g] >= lb * y[b], name=f'blend_lb_{b}_{g}')
for (b, g), ub in U_bg.items():
    m.addConstr(x[b, g] <= ub * y[b], name=f'blend_ub_{b}_{g}')
for g in G:
    m.addConstr(gp.quicksum((x[b, g] for b in B)) <= S_g[g], name=f'supply_{g}')
if 'Red' not in B:
    raise ValueError("Brand 'Red' not found in brand list.")
m.addConstr(y['Red'] >= 2000, name='minprod_Red')
revenue = gp.quicksum((P_b[b] * y[b] for b in B))
cost = gp.quicksum((C_g[g] * gp.quicksum((x[b, g] for b in B)) for g in G))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')