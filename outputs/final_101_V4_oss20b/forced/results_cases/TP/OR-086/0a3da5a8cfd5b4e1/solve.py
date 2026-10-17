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
             'role': 'descriptive non-unique role',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    tables = CSVQA_DATA['tables']
    s_g = {}
    c_g = {}
    p_b = {}
    G = set()
    B = set()
    for table in tables:
        if table['table_id'] == 'file_0_view_0':
            for record in table['records']:
                values = record['values']
                grade = values['Grade'].strip()
                G.add(grade)
                try:
                    s_g[grade] = float(values['Daily Supply (kg)'])
                except (KeyError, ValueError):
                    raise ValueError(f'Missing or invalid Daily Supply for grade {grade}')
                try:
                    c_g[grade] = float(values['Cost (CNY/kg)'])
                except (KeyError, ValueError):
                    raise ValueError(f'Missing or invalid Cost for grade {grade}')
    for table in tables:
        if table['table_id'] == 'file_1_view_0':
            for record in table['records']:
                values = record['values']
                brand = values['Brand'].strip()
                B.add(brand)
                try:
                    p_b[brand] = float(values['Selling Price (CNY/kg)'])
                except (KeyError, ValueError):
                    raise ValueError(f'Missing or invalid Selling Price for brand {brand}')
    for g in G:
        if g not in s_g:
            raise ValueError(f'Supply limit missing for grade {g}')
        if g not in c_g:
            raise ValueError(f'Cost missing for grade {g}')
    for b in B:
        if b not in p_b:
            raise ValueError(f'Selling price missing for brand {b}')
    l = {}
    u = {}
    for g in G:
        for b in B:
            l[g, b] = 0.0
            u[g, b] = 1.0
    for table in tables:
        if table['table_id'] == 'file_1_view_0':
            for record in table['records']:
                values = record['values']
                brand = values['Brand'].strip()
                req_str = values['Blending Requirements']
                clauses = re.findall('(\\w+)\\s+(less than|more than)\\s+(\\d+)%', req_str)
                for grade, relation, percent_str in clauses:
                    grade = grade.strip()
                    if grade not in G:
                        raise ValueError(f'Grade {grade} in blending requirements not defined')
                    percent = float(percent_str) / 100.0
                    key = (grade, brand)
                    if relation == 'less than':
                        u[key] = percent
                    elif relation == 'more than':
                        l[key] = percent
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    var_keys = [(g, b) for g in G for b in B]
    x = model.addVars(var_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    revenue = gurobipy.quicksum((p_b[b] * gurobipy.quicksum((x[g, b] for g in G)) for b in B))
    cost = gurobipy.quicksum((c_g[g] * gurobipy.quicksum((x[g, b] for b in B)) for g in G))
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    for g in G:
        model.addConstr(gurobipy.quicksum((x[g, b] for b in B)) <= s_g[g], name=f'supply_{g}')
    for g in G:
        for b in B:
            total_b = gurobipy.quicksum((x[g_prime, b] for g_prime in G))
            model.addConstr(l[g, b] * total_b <= x[g, b], name=f'blend_lower_{g}_{b}')
            model.addConstr(x[g, b] <= u[g, b] * total_b, name=f'blend_upper_{g}_{b}')
    if 'Red' in B:
        model.addConstr(gurobipy.quicksum((x[g, 'Red'] for g in G)) >= 2000, name='min_red')
    else:
        raise ValueError("Brand 'Red' not defined in data")
    model.optimize()
    return model
m = solve_problem()