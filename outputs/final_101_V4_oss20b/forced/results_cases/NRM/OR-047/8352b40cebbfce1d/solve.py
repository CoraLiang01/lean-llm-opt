CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A digital game store needs to decide which games to list on different platforms, considering that these '
          'games belong to various genres such as racing, sports, and others. Each platform has a limited memory '
          'capacity, with specific details provided in "capacity.csv." The predefined value and memory requirement of '
          'each game are available in "products.csv." The objective is to determine which genres and how many units of '
          'each game to list on each platform to maximize the total value of the games across all platforms, while '
          'ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  '
          'x_ij represent the number of units of games from genres j to be listed on platform i.The decision variables '
          'must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['PlatformId', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '1336', 'PlatformId': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '1754', 'PlatformId': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '1617', 'PlatformId': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '1119', 'PlatformId': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '1410', 'PlatformId': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '627', 'PlatformId': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '748', 'PlatformId': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '1540', 'PlatformId': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '1292', 'PlatformId': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '1138', 'PlatformId': '10'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Racing', 'Value': '28', 'Weight': '393'}},
                         {'source_row': 1, 'values': {'ProductName': 'Sports', 'Value': '69', 'Weight': '195'}},
                         {'source_row': 2, 'values': {'ProductName': 'Action', 'Value': '20', 'Weight': '192'}},
                         {'source_row': 3, 'values': {'ProductName': 'Adventure', 'Value': '62', 'Weight': '155'}},
                         {'source_row': 4, 'values': {'ProductName': 'RPG', 'Value': '58', 'Weight': '500'}},
                         {'source_row': 5, 'values': {'ProductName': 'Shooter', 'Value': '11', 'Weight': '156'}},
                         {'source_row': 6, 'values': {'ProductName': 'Strategy', 'Value': '73', 'Weight': '317'}},
                         {'source_row': 7, 'values': {'ProductName': 'Simulation', 'Value': '43', 'Weight': '694'}},
                         {'source_row': 8, 'values': {'ProductName': 'Puzzle', 'Value': '28', 'Weight': '751'}},
                         {'source_row': 9, 'values': {'ProductName': 'Fighting', 'Value': '57', 'Weight': '467'}},
                         {'source_row': 10, 'values': {'ProductName': 'Platformer', 'Value': '92', 'Weight': '796'}},
                         {'source_row': 11, 'values': {'ProductName': 'Survival', 'Value': '66', 'Weight': '146'}},
                         {'source_row': 12, 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '269'}},
                         {'source_row': 13, 'values': {'ProductName': 'Sandbox', 'Value': '49', 'Weight': '246'}},
                         {'source_row': 14, 'values': {'ProductName': 'MMO', 'Value': '12', 'Weight': '652'}}],
             'returned_rows': 15,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    C_p = {}
    for rec in cap_table['records']:
        p = rec['values']['PlatformId']
        cap = int(rec['values']['Capacity'])
        C_p[p] = cap
    v_g = {}
    w_g = {}
    for rec in prod_table['records']:
        g = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        v_g[g] = val
        w_g[g] = wt
    P = list(C_p.keys())
    G = list(v_g.keys())
    for p in P:
        if p not in C_p:
            raise ValueError(f'Missing capacity for platform {p}')
    for g in G:
        if g not in v_g or g not in w_g:
            raise ValueError(f'Missing value or weight for genre {g}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(p, g) for p in P for g in G], vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((v_g[g] * x[p, g] for p in P for g in G)), GRB.MAXIMIZE)
    for p in P:
        model.addConstr(gurobipy.quicksum((w_g[g] * x[p, g] for g in G)) <= C_p[p], name='')
    model.optimize()
    return model
m = solve_problem()