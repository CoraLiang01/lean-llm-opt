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
data = CSVQA_DATA
table_0 = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
grades = []
S_g = {}
C_g = {}
for rec in table_0['records']:
    g = rec['values']['Grade']
    grades.append(g)
    try:
        S_g[g] = float(rec['values']['Daily Supply (kg)'])
        C_g[g] = float(rec['values']['Cost (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-1.csv for grade {g}: {e}')
table_1 = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
brands = []
P_b = {}
L_gb = {g: {} for g in grades}
U_gb = {g: {} for g in grades}
for rec in table_1['records']:
    b = rec['values']['Brand']
    brands.append(b)
    try:
        P_b[b] = float(rec['values']['Selling Price (CNY/kg)'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in 30-2.csv for brand {b}: {e}')
    blend = rec['values']['Blending Requirements']
    pattern = '([A-Za-z0-9]+)\\s+(less than|more than)\\s+([0-9]+)%'
    matches = re.findall(pattern, blend)
    if not matches and blend.strip():
        raise ValueError(f"Unmatched blending requirement clause in brand {b}: '{blend}'")
    for (g_req, sense, pct) in matches:
        if g_req not in grades:
            raise ValueError(f"Unknown grade '{g_req}' in blending requirements for brand {b}")
        prop = float(pct) / 100.0
        if sense == 'less than':
            U_gb[g_req][b] = prop
        elif sense == 'more than':
            L_gb[g_req][b] = prop
        else:
            raise ValueError(f"Unknown sense '{sense}' in blending requirements for brand {b}")
if set(grades) != set(L_gb.keys()) or set(grades) != set(U_gb.keys()):
    raise ValueError('Mismatch in grade keys between supply/cost and blending requirements.')
m = gp.Model('wine_blending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'ydef_{b}')
for b in brands:
    for g in grades:
        if b in L_gb[g]:
            m.addConstr(x_vars[g, b] >= L_gb[g][b] * y_vars[b], name=f'blend_lb_{g}_{b}')
        if b in U_gb[g]:
            m.addConstr(x_vars[g, b] <= U_gb[g][b] * y_vars[b], name=f'blend_ub_{g}_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= S_g[g], name=f'supply_{g}')
if 'Red' not in brands:
    raise ValueError("Brand 'Red' not found in brands list.")
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
revenue = gp.quicksum((P_b[b] * y_vars[b] for b in brands))
cost = gp.quicksum((C_g[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')