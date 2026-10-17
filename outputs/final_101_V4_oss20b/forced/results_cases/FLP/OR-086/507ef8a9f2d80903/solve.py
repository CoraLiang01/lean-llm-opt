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
 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'],
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
            {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'],
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
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    G = [rec['values']['Grade'] for rec in table0['records']]
    B = [rec['values']['Brand'] for rec in table1['records']]
    s_g = {}
    c_g = {}
    for rec in table0['records']:
        grade = rec['values']['Grade']
        try:
            s_g[grade] = float(rec['values']['Daily Supply (kg)'])
            c_g[grade] = float(rec['values']['Cost (CNY/kg)'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid supply/cost for grade {grade}') from e
    p_b = {}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        try:
            p_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid selling price for brand {brand}') from e
    l_b_g = {}
    u_b_g = {}
    for b in B:
        for g in G:
            l_b_g[b, g] = 0.0
            u_b_g[b, g] = 1.0
    pattern = re.compile('([A-Za-z]+)\\s+(less than|more than)\\s+([\\d.]+)%')
    for rec in table1['records']:
        brand = rec['values']['Brand']
        blending = rec['values']['Blending Requirements']
        for match in pattern.findall(blending):
            grade, bound_type, percent_str = match
            percent = float(percent_str) / 100.0
            if bound_type == 'less than':
                u_b_g[brand, grade] = percent
            else:
                l_b_g[brand, grade] = percent
    for g in G:
        if g not in s_g:
            raise ValueError(f'Supply limit missing for grade {g}')
        if g not in c_g:
            raise ValueError(f'Cost missing for grade {g}')
    for b in B:
        if b not in p_b:
            raise ValueError(f'Selling price missing for brand {b}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(b, g) for b in B for g in G], lb=0, vtype=GRB.CONTINUOUS, name='')
    for b in B:
        X_b_expr = gurobipy.quicksum((x[b, g] for g in G))
        for g in G:
            model.addConstr(l_b_g[b, g] * X_b_expr <= x[b, g], name=f'lb_{b}_{g}')
            model.addConstr(x[b, g] <= u_b_g[b, g] * X_b_expr, name=f'ub_{b}_{g}')
    for g in G:
        model.addConstr(gurobipy.quicksum((x[b, g] for b in B)) <= s_g[g], name=f'supply_{g}')
    X_red_expr = gurobipy.quicksum((x['Red', g] for g in G))
    model.addConstr(X_red_expr >= 2000, name='min_red')
    revenue_expr = gurobipy.quicksum((p_b[b] * gurobipy.quicksum((x[b, g] for g in G)) for b in B))
    cost_expr = gurobipy.quicksum((c_g[g] * gurobipy.quicksum((x[b, g] for b in B)) for g in G))
    model.setObjective(revenue_expr - cost_expr, GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()