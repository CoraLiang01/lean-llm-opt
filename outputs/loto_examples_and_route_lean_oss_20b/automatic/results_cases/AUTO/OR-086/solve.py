CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB
import re
tables = CSVQA_DATA['tables']
grade_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
grades = []
supply = {}
cost = {}
for rec in grade_table['records']:
    g = rec['values']['Grade'].strip()
    grades.append(g)
    supply[g] = float(rec['values']['Daily Supply (kg)'])
    cost[g] = float(rec['values']['Cost (CNY/kg)'])
brand_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
brands = []
price = {}
blending_bounds = {}
for rec in brand_table['records']:
    b = rec['values']['Brand'].strip()
    brands.append(b)
    price[b] = float(rec['values']['Selling Price (CNY/kg)'])
    req_text = rec['values']['Blending Requirements']
    clauses = re.findall('([A-Z]+)\\s+(less than|more than)\\s+(\\d+)%', req_text)
    for grade, op, pct in clauses:
        key = (grade.strip(), b)
        pct_val = float(pct) / 100.0
        if key not in blending_bounds:
            blending_bounds[key] = [None, None]
        if op == 'less than':
            blending_bounds[key][1] = pct_val
        else:
            blending_bounds[key][0] = pct_val
m = gurobipy.Model('wine_blending')
m.setParam('MIPGap', 1e-06)
x = m.addVars([(g, b) for g in grades for b in brands], lb=0, vtype=GRB.CONTINUOUS, name='')
P = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(P[b] == gurobipy.quicksum((x[g, b] for g in grades)), name=f'P_def_{b}')
for (g, b), bounds in blending_bounds.items():
    lower, upper = bounds
    if lower is not None:
        m.addConstr(x[g, b] >= lower * P[b], name=f'lower_{g}_{b}')
    if upper is not None:
        m.addConstr(x[g, b] <= upper * P[b], name=f'upper_{g}_{b}')
for g in grades:
    m.addConstr(gurobipy.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
m.addConstr(P['Red'] >= 2000, name='min_red')
obj = gurobipy.quicksum((price[b] * P[b] - gurobipy.quicksum((cost[g] * x[g, b] for g in grades)) for b in brands))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
