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
 'tables': [{'columns': ['resource_id', 'resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'resource_capacity': '1336', 'resource_id': '1'}},
                         {'source_row': 1, 'values': {'resource_capacity': '1754', 'resource_id': '2'}},
                         {'source_row': 2, 'values': {'resource_capacity': '1617', 'resource_id': '3'}},
                         {'source_row': 3, 'values': {'resource_capacity': '1119', 'resource_id': '4'}},
                         {'source_row': 4, 'values': {'resource_capacity': '1410', 'resource_id': '5'}},
                         {'source_row': 5, 'values': {'resource_capacity': '627', 'resource_id': '6'}},
                         {'source_row': 6, 'values': {'resource_capacity': '748', 'resource_id': '7'}},
                         {'source_row': 7, 'values': {'resource_capacity': '1540', 'resource_id': '8'}},
                         {'source_row': 8, 'values': {'resource_capacity': '1292', 'resource_id': '9'}},
                         {'source_row': 9, 'values': {'resource_capacity': '1138', 'resource_id': '10'}}],
             'returned_rows': 10,
             'role': 'platform resource capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['item_name', 'item_value', 'resource_requirement'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'item_name': 'Racing', 'item_value': '28', 'resource_requirement': '393'}},
                         {'source_row': 1,
                          'values': {'item_name': 'Sports', 'item_value': '69', 'resource_requirement': '195'}},
                         {'source_row': 2,
                          'values': {'item_name': 'Action', 'item_value': '20', 'resource_requirement': '192'}},
                         {'source_row': 3,
                          'values': {'item_name': 'Adventure', 'item_value': '62', 'resource_requirement': '155'}},
                         {'source_row': 4,
                          'values': {'item_name': 'RPG', 'item_value': '58', 'resource_requirement': '500'}},
                         {'source_row': 5,
                          'values': {'item_name': 'Shooter', 'item_value': '11', 'resource_requirement': '156'}},
                         {'source_row': 6,
                          'values': {'item_name': 'Strategy', 'item_value': '73', 'resource_requirement': '317'}},
                         {'source_row': 7,
                          'values': {'item_name': 'Simulation', 'item_value': '43', 'resource_requirement': '694'}},
                         {'source_row': 8,
                          'values': {'item_name': 'Puzzle', 'item_value': '28', 'resource_requirement': '751'}},
                         {'source_row': 9,
                          'values': {'item_name': 'Fighting', 'item_value': '57', 'resource_requirement': '467'}},
                         {'source_row': 10,
                          'values': {'item_name': 'Platformer', 'item_value': '92', 'resource_requirement': '796'}},
                         {'source_row': 11,
                          'values': {'item_name': 'Survival', 'item_value': '66', 'resource_requirement': '146'}},
                         {'source_row': 12,
                          'values': {'item_name': 'Horror', 'item_value': '14', 'resource_requirement': '269'}},
                         {'source_row': 13,
                          'values': {'item_name': 'Sandbox', 'item_value': '49', 'resource_requirement': '246'}},
                         {'source_row': 14,
                          'values': {'item_name': 'MMO', 'item_value': '12', 'resource_requirement': '652'}}],
             'returned_rows': 15,
             'role': 'game genre and value',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    platforms_df = CSVQA_FRAMES['file_0_view_0']
    genres_df = CSVQA_FRAMES['file_1_view_0']
    platforms = list(platforms_df['resource_id'])
    genres = list(genres_df['item_name'])
    capacity = {}
    for (idx, row) in platforms_df.iterrows():
        pid = row['resource_id']
        try:
            cap = int(row['resource_capacity'])
        except Exception:
            raise ValueError(f"Invalid resource_capacity for platform {pid}: {row['resource_capacity']}")
        capacity[pid] = cap
    value = {}
    requirement = {}
    for (idx, row) in genres_df.iterrows():
        gid = row['item_name']
        try:
            val = int(row['item_value'])
        except Exception:
            raise ValueError(f"Invalid item_value for genre {gid}: {row['item_value']}")
        try:
            req = int(row['resource_requirement'])
        except Exception:
            raise ValueError(f"Invalid resource_requirement for genre {gid}: {row['resource_requirement']}")
        value[gid] = val
        requirement[gid] = req
    if set(platforms) != set(capacity.keys()):
        raise ValueError('Mismatch in platform IDs and capacity keys')
    if set(genres) != set(value.keys()) or set(genres) != set(requirement.keys()):
        raise ValueError('Mismatch in genre IDs and value/requirement keys')
    m = gp.Model('Game_Store_Listing')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((requirement[j] * quantity_vars[i, j] for j in genres)) <= capacity[i] for i in platforms), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')