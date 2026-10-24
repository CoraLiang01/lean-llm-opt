CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import re
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA['tables']
table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
grades = [rec['values']['Grade'] for rec in table0['records']]
S_g = {rec['values']['Grade']: float(rec['values']['Daily Supply (kg)']) for rec in table0['records']}
c_g = {rec['values']['Grade']: float(rec['values']['Cost (CNY/kg)']) for rec in table0['records']}
brands = [rec['values']['Brand'] for rec in table1['records']]
p_b = {rec['values']['Brand']: float(rec['values']['Selling Price (CNY/kg)']) for rec in table1['records']}
l_bounds = {(g, b): 0.0 for g in grades for b in brands}
u_bounds = {(g, b): 1.0 for g in grades for b in brands}
for rec in table1['records']:
    brand = rec['values']['Brand']
    req = rec['values']['Blending Requirements']
    for match in re.findall('([A-Z]+)\\s+less\\s+than\\s+(\\d+)%', req):
        g, pct = match
        u_bounds[g, brand] = float(pct) / 100.0
    for match in re.findall('([A-Z]+)\\s+more\\s+than\\s+(\\d+)%', req):
        g, pct = match
        l_bounds[g, brand] = float(pct) / 100.0

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in grades for b in brands], lb=0, vtype=GRB.CONTINUOUS, name='')
    for b in brands:
        total_b = gurobipy.quicksum((x[h, b] for h in grades))
        for g in grades:
            m.addConstr(x[g, b] >= l_bounds[g, b] * total_b, name=f'lb_{g}_{b}')
            m.addConstr(x[g, b] <= u_bounds[g, b] * total_b, name=f'ub_{g}_{b}')
    for g in grades:
        m.addConstr(gurobipy.quicksum((x[g, b] for b in brands)) <= S_g[g], name=f'supply_{g}')
    m.addConstr(gurobipy.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='min_red')
    revenue = gurobipy.quicksum((p_b[b] * gurobipy.quicksum((x[g, b] for g in grades)) for b in brands))
    cost = gurobipy.quicksum((c_g[g] * gurobipy.quicksum((x[g, b] for b in brands)) for g in grades))
    m.setObjective(revenue - cost, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
