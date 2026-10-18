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
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
table0 = CSVQA_DATA['tables'][0]['records']
table1 = CSVQA_DATA['tables'][1]['records']
df0 = pd.DataFrame([rec['values'] for rec in table0])
df1 = pd.DataFrame([rec['values'] for rec in table1])
df0['Daily Supply (kg)'] = pd.to_numeric(df0['Daily Supply (kg)'])
df0['Cost (CNY/kg)'] = pd.to_numeric(df0['Cost (CNY/kg)'])
df1['Selling Price (CNY/kg)'] = pd.to_numeric(df1['Selling Price (CNY/kg)'])
grades = df0['Grade'].tolist()
brands = df1['Brand'].tolist()
S = dict(zip(grades, df0['Daily Supply (kg)']))
C = dict(zip(grades, df0['Cost (CNY/kg)']))
P = dict(zip(brands, df1['Selling Price (CNY/kg)']))
L = {(b, g): 0 for b in brands for g in grades}
U = {(b, g): 1 for b in brands for g in grades}
for _, row in df1.iterrows():
    brand = row['Brand']
    req = row['Blending Requirements']
    for m in re.finditer('([A-Z]+)\\s+(less than|more than)\\s+(\\d+)%', req):
        grade = m.group(1)
        cond = m.group(2)
        val = int(m.group(3)) / 100.0
        if cond == 'less than':
            U[brand, grade] = min(U[brand, grade], val)
        else:
            L[brand, grade] = max(L[brand, grade], val)
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {(b, g): m.addVar(vtype=GRB.CONTINUOUS) for b in brands for g in grades}
y = {b: m.addVar(vtype=GRB.CONTINUOUS) for b in brands}
for b in brands:
    m.addConstr(y[b] == gp.quicksum((x[b, g] for g in grades)))
for b in brands:
    for g in grades:
        m.addConstr(x[b, g] >= L[b, g] * y[b])
        m.addConstr(x[b, g] <= U[b, g] * y[b])
for g in grades:
    m.addConstr(gp.quicksum((x[b, g] for b in brands)) <= S[g])
m.addConstr(y['Red'] >= 2000)
m.setObjective(gp.quicksum((P[b] * y[b] for b in brands)) - gp.quicksum((C[g] * x[b, g] for b in brands for g in grades)), GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.status)