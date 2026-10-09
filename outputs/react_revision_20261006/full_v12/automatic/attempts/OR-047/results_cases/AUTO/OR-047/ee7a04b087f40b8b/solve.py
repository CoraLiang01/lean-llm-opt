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
 'route': 'RA',
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
             'role': 'platform capacity',
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
             'role': 'game genre value and memory',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    platform_frame = CSVQA_FRAMES['file_0_view_0']
    platforms = []
    capacity = {}
    for (_, row) in platform_frame.iterrows():
        pid = row['PlatformId']
        platforms.append(pid)
        capacity[pid] = float(row['Capacity'])
    product_frame = CSVQA_FRAMES['file_1_view_0']
    genres = []
    value = {}
    weight = {}
    for (_, row) in product_frame.iterrows():
        pname = row['ProductName']
        genres.append(pname)
        value[pname] = float(row['Value'])
        weight[pname] = float(row['Weight'])
    m = gp.Model('game_store_allocation')
    x_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in genres)) <= capacity[i] for i in platforms), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')