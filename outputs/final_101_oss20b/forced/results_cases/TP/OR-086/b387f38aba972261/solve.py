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
 'route': 'TP',
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
import json
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
data = CSVQA_DATA
table0 = data['tables'][0]
table1 = data['tables'][1]
df0 = pd.DataFrame([rec['values'] for rec in table0['records']])
df1 = pd.DataFrame([rec['values'] for rec in table1['records']])
df0['Daily Supply (kg)'] = df0['Daily Supply (kg)'].astype(float)
df0['Cost (CNY/kg)'] = df0['Cost (CNY/kg)'].astype(float)
df1['Selling Price (CNY/kg)'] = df1['Selling Price (CNY/kg)'].astype(float)
grades = df0['Grade'].tolist()
brands = df1['Brand'].tolist()
supply_limit = dict(zip(df0['Grade'], df0['Daily Supply (kg)']))
cost = dict(zip(df0['Grade'], df0['Cost (CNY/kg)']))
selling_price = dict(zip(df1['Brand'], df1['Selling Price (CNY/kg)']))
blending_req = {}
for _, row in df1.iterrows():
    brand = row['Brand']
    req_str = row['Blending Requirements']
    constraints = []
    parts = re.split('\\s{2,}', req_str.strip())
    for part in parts:
        m = re.match('([A-Z])\\s+(less than|more than)\\s+(\\d+)%', part.strip())
        if m:
            grade = m.group(1)
            bound_type = m.group(2)
            percent = int(m.group(3)) / 100.0
            constraints.append((grade, bound_type, percent))
    blending_req[brand] = constraints
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {}
for g in grades:
    for b in brands:
        var = m.addVar(vtype=GRB.CONTINUOUS)
        x[g, b] = var
m.update()
obj = gp.quicksum(((selling_price[b] - cost[g]) * x[g, b] for g in grades for b in brands))
m.setObjective(obj, GRB.MAXIMIZE)
for b in brands:
    total_b = gp.quicksum((x[g, b] for g in grades))
    for g, bound_type, percent in blending_req[b]:
        if bound_type == 'less than':
            m.addConstr(x[g, b] <= percent * total_b)
        else:
            m.addConstr(x[g, b] >= percent * total_b)
for g in grades:
    total_g = gp.quicksum((x[g, b] for b in brands))
    m.addConstr(total_g <= supply_limit[g])
total_red = gp.quicksum((x[g, 'Red'] for g in grades))
m.addConstr(total_red >= 2000)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Status: {m.status}')