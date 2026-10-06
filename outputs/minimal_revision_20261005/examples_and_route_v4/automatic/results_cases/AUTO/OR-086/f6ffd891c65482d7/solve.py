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

def solve_problem():
    grades = []
    S_g = {}
    C_g = {}
    for rec in CSVQA_DATA['tables'][0]['records']:
        g = rec['values']['Grade']
        grades.append(g)
        S_g[g] = float(rec['values']['Daily Supply (kg)'])
        C_g[g] = float(rec['values']['Cost (CNY/kg)'])
    brands = []
    P_b = {}
    L = {}
    U = {}
    for rec in CSVQA_DATA['tables'][1]['records']:
        b = rec['values']['Brand']
        brands.append(b)
        P_b[b] = float(rec['values']['Selling Price (CNY/kg)'])
        req = rec['values']['Blending Requirements']
        reqs = req.split()
        i = 0
        while i < len(reqs):
            if reqs[i] in grades:
                g = reqs[i]
                if i + 2 < len(reqs):
                    if reqs[i + 1].casefold() == 'less' and reqs[i + 2].casefold().startswith('than'):
                        percent = float(reqs[i + 2][4:-1] if reqs[i + 2].endswith('%') else reqs[i + 2][4:]) / 100.0
                        U[g, b] = percent
                        i += 3
                        continue
                    elif reqs[i + 1].casefold() == 'more' and reqs[i + 2].casefold().startswith('than'):
                        percent = float(reqs[i + 2][4:-1] if reqs[i + 2].endswith('%') else reqs[i + 2][4:]) / 100.0
                        L[g, b] = percent
                        i += 3
                        continue
            i += 1
    m = gp.Model('wine_blending')
    m.Params.MIPGap = 0.0001
    x_keys = [(g, b) for g in grades for b in brands]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.addConstrs((y[b] == gp.quicksum((x[g, b] for g in grades)) for b in brands), name='')
    m.setObjective(gp.quicksum((P_b[b] * y[b] for b in brands)) - gp.quicksum((C_g[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((x[g, b] for b in brands)) <= S_g[g] for g in grades), name='')
    for ((g, b), lb) in L.items():
        m.addConstr(x[g, b] >= lb * y[b], name=f'blend_lb_{g}_{b}')
    for ((g, b), ub) in U.items():
        m.addConstr(x[g, b] <= ub * y[b], name=f'blend_ub_{g}_{b}')
    if 'Red' in brands:
        m.addConstr(y['Red'] >= 2000, name='min_red')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')