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
             'role': 'file_0',
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
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'PlatformId', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'PlatformId'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'PlatformId', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'PlatformId'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    # Read data from CSVQA_FRAMES
    capacity_frame = CSVQA_FRAMES["file_0_view_0"]
    products_frame = CSVQA_FRAMES["file_1_view_0"]

    # Index sets
    platforms = []
    c = {}
    for _, row in capacity_frame.iterrows():
        platform_id = row["PlatformId"]
        platforms.append(platform_id)
        c[platform_id] = float(row["Capacity"])

    genres = []
    v = {}
    w = {}
    for _, row in products_frame.iterrows():
        genre = row["ProductName"]
        genres.append(genre)
        v[genre] = float(row["Value"])
        w[genre] = float(row["Weight"])

    # Model
    m = gp.Model("game_store_allocation")

    # Decision variables: x_{ij} >= 0, integer
    x_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name="x")

    # Objective: maximize total value
    m.setObjective(
        gp.quicksum(v[j] * x_vars[i, j] for i in platforms for j in genres),
        GRB.MAXIMIZE
    )

    # Platform memory capacity constraints
    m.addConstrs(
        (gp.quicksum(w[j] * x_vars[i, j] for j in genres) <= c[i] for i in platforms),
        name="capacity"
    )

    m.Params.MIPGap = 1e-4
    m.optimize()
    return m

m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')