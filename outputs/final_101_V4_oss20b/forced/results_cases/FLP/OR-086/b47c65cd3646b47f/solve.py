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
 'route': 'FLP',
 'tables': [{'aliases': ['file_0_view_0', '30-1.csv'],
             'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'],
             'file_index': 0,
             'file_name': '30-1.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}},
                         {'source_row': 1,
                          'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}},
                         {'source_row': 2,
                          'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', '30-2.csv'],
             'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'],
             'file_index': 1,
             'file_name': '30-2.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
import re
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
grade_index = {g: i for i, g in enumerate(grades)}
brand_index = {b: j for j, b in enumerate(brands)}
d = [0.0] * len(grades)
c = [0.0] * len(grades)
p = [0.0] * len(brands)
L = [[0.0] * len(brands) for _ in range(len(grades))]
U = [[0.0] * len(brands) for _ in range(len(grades))]
for rec in CSVQA_DATA['tables'][0]['records']:
    grade = rec['values']['Grade']
    d[grade_index[grade]] = float(rec['values']['Daily Supply (kg)'])
    c[grade_index[grade]] = float(rec['values']['Cost (CNY/kg)'])
for rec in CSVQA_DATA['tables'][1]['records']:
    brand = rec['values']['Brand']
    p[brand_index[brand]] = float(rec['values']['Selling Price (CNY/kg)'])
    req_str = rec['values']['Blending Requirements']
    for match in re.finditer('([A-Z]+)\\s+(less than|more than)\\s+(\\d+)%', req_str):
        g = match.group(1)
        relation = match.group(2)
        percent = float(match.group(3)) / 100.0
        i = grade_index[g]
        j = brand_index[brand]
        if relation == 'less than':
            U[i][j] = percent
        else:
            L[i][j] = percent
m = gp.Model('Wine_Production')
x = m.addVars(len(grades), len(brands), lb=0)
obj = gp.quicksum((p[j] * x[i, j] for i in range(len(grades)) for j in range(len(brands)))) - gp.quicksum((c[i] * x[i, j] for i in range(len(grades)) for j in range(len(brands))))
m.setObjective(obj, GRB.MAXIMIZE)
for i in range(len(grades)):
    m.addConstr(gp.quicksum((x[i, j] for j in range(len(brands)))) <= d[i])
m.addConstr(gp.quicksum((x[i, brand_index['Red']] for i in range(len(grades)))) >= 2000)
for i in range(len(grades)):
    for j in range(len(brands)):
        total_j = gp.quicksum((x[k, j] for k in range(len(grades))))
        if L[i][j] > 0:
            m.addConstr(L[i][j] * total_j <= x[i, j])
        if U[i][j] > 0:
            m.addConstr(x[i, j] <= U[i][j] * total_j)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)