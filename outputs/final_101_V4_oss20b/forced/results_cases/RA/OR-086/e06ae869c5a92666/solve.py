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
 'route': 'RA',
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
             'table_id': 'file_1_view_0'},
            {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'],
             'file_index': 1,
             'file_name': '30-2.csv',
             'filters': {'conditions': [{'column': 'Brand',
                                         'dtype': 'string',
                                         'evidence': 'Red brand must have a daily production level of at least 2,000 '
                                                     'kg',
                                         'inclusive': 'both',
                                         'operator': 'eq',
                                         'value': 'Red'}],
                         'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Blending Requirements': 'I less than 10%  II more than 50%',
                                     'Brand': 'Red',
                                     'Selling Price (CNY/kg)': '5.5'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_1'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    tables = CSVQA_DATA['tables']
    supply_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    brand_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    S = {}
    c = {}
    G = set()
    for rec in supply_table['records']:
        grade = rec['values']['Grade']
        G.add(grade)
        S[grade] = float(rec['values']['Daily Supply (kg)'])
        c[grade] = float(rec['values']['Cost (CNY/kg)'])
    p = {}
    B = set()
    l = {}
    u = {}
    for rec in brand_table['records']:
        brand = rec['values']['Brand']
        B.add(brand)
        p[brand] = float(rec['values']['Selling Price (CNY/kg)'])
        req_str = rec['values']['Blending Requirements']
        local_l = {}
        local_u = {}
        for m in re.finditer('([A-Z]+)\\s+(less than|more than)\\s+(\\d+)%', req_str):
            grade, relation, percent = m.groups()
            val = int(percent) / 100.0
            if relation == 'less than':
                local_u[grade] = val
            else:
                local_l[grade] = val
        for g in G:
            l[g, brand] = local_l.get(g, 0.0)
            u[g, brand] = local_u.get(g, 1.0)
    for g in G:
        if g not in S or g not in c:
            raise ValueError(f'Missing supply or cost for grade {g}')
    for b in B:
        if b not in p:
            raise ValueError(f'Missing selling price for brand {b}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in G for b in B], lb=0, vtype=GRB.CONTINUOUS, name='')
    revenue = gurobipy.quicksum((p[b] * x[g, b] for g in G for b in B))
    cost = gurobipy.quicksum((c[g] * x[g, b] for g in G for b in B))
    m.setObjective(revenue - cost, GRB.MAXIMIZE)
    for g in G:
        m.addConstr(gurobipy.quicksum((x[g, b] for b in B)) <= S[g], name=f'supply_{g}')
    for g in G:
        for b in B:
            total_b = gurobipy.quicksum((x[h, b] for h in G))
            m.addConstr(l[g, b] * total_b <= x[g, b], name=f'lb_{g}_{b}')
            m.addConstr(x[g, b] <= u[g, b] * total_b, name=f'ub_{g}_{b}')
    m.addConstr(gurobipy.quicksum((x[g, 'Red'] for g in G)) >= 2000, name='red_min')
    m.optimize()
    return m
m = solve_problem()