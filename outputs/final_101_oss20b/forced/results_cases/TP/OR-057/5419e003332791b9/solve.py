CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A digital game store needs to decide which games to list on different platforms, considering that these '
          'games belong to various genres such as racing, sports, and others. Each platform has a limited memory '
          'capacity, with specific details provided in "capacity.csv." The predefined value and memory requirement of '
          'each game are available in "products.csv." The objective is to determine which genres and how many units of '
          'each game to list on each platform to maximize the total value of the games across all platforms, while '
          'ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  '
          'x_ij represent the number of units of game j to be listed on platform i.The decision variables must be '
          'integers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'capacity.csv'],
             'columns': ['PlatformID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'Capacity': '995', 'PlatformID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '1143', 'PlatformID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '949', 'PlatformID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '969', 'PlatformID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '1649', 'PlatformID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '870', 'PlatformID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '1064', 'PlatformID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '536', 'PlatformID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '766', 'PlatformID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '532', 'PlatformID': '10'}},
                         {'source_row': 10, 'values': {'Capacity': '1703', 'PlatformID': '11'}},
                         {'source_row': 11, 'values': {'Capacity': '1633', 'PlatformID': '12'}},
                         {'source_row': 12, 'values': {'Capacity': '1203', 'PlatformID': '13'}},
                         {'source_row': 13, 'values': {'Capacity': '1979', 'PlatformID': '14'}},
                         {'source_row': 14, 'values': {'Capacity': '1797', 'PlatformID': '15'}}],
             'returned_rows': 15,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'products.csv'],
             'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Racing', 'Value': '59', 'Weight': '776'}},
                         {'source_row': 1, 'values': {'ProductName': 'Sports', 'Value': '83', 'Weight': '573'}},
                         {'source_row': 2, 'values': {'ProductName': 'Action', 'Value': '94', 'Weight': '127'}},
                         {'source_row': 3, 'values': {'ProductName': 'Adventure', 'Value': '41', 'Weight': '138'}},
                         {'source_row': 4, 'values': {'ProductName': 'RPG', 'Value': '96', 'Weight': '385'}},
                         {'source_row': 5, 'values': {'ProductName': 'Shooter', 'Value': '12', 'Weight': '263'}},
                         {'source_row': 6, 'values': {'ProductName': 'Strategy', 'Value': '83', 'Weight': '473'}},
                         {'source_row': 7, 'values': {'ProductName': 'Simulation', 'Value': '36', 'Weight': '387'}},
                         {'source_row': 8, 'values': {'ProductName': 'Puzzle', 'Value': '56', 'Weight': '390'}},
                         {'source_row': 9, 'values': {'ProductName': 'Fighting', 'Value': '27', 'Weight': '556'}},
                         {'source_row': 10, 'values': {'ProductName': 'Platformer', 'Value': '47', 'Weight': '601'}},
                         {'source_row': 11, 'values': {'ProductName': 'Survival', 'Value': '24', 'Weight': '441'}},
                         {'source_row': 12, 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '603'}},
                         {'source_row': 13, 'values': {'ProductName': 'Sandbox', 'Value': '22', 'Weight': '411'}},
                         {'source_row': 14, 'values': {'ProductName': 'MMO', 'Value': '17', 'Weight': '652'}}],
             'returned_rows': 15,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id_or_alias):
    for tbl in tables:
        if tbl['table_id'] == table_id_or_alias or table_id_or_alias in tbl.get('aliases', []):
            return tbl
    raise KeyError(f'Table {table_id_or_alias} not found')
cap_tbl = get_table('file_0_view_0')
platform_ids = []
platform_cap = {}
for rec in cap_tbl['records']:
    pid = rec['values']['PlatformID']
    cap = int(rec['values']['Capacity'])
    platform_ids.append(pid)
    platform_cap[pid] = cap
prod_tbl = get_table('file_1_view_0')
product_names = []
product_value = {}
product_weight = {}
for rec in prod_tbl['records']:
    pname = rec['values']['ProductName']
    val = int(rec['values']['Value'])
    wt = int(rec['values']['Weight'])
    product_names.append(pname)
    product_value[pname] = val
    product_weight[pname] = wt
m = gp.Model('game_listing')
m.Params.MIPGap = 1e-06
x = {}
for i in platform_ids:
    for j in product_names:
        var = m.addVar(vtype=GRB.INTEGER)
        x[i, j] = var
for i in platform_ids:
    expr = gp.quicksum((product_weight[j] * x[i, j] for j in product_names))
    m.addConstr(expr <= platform_cap[i])
obj = gp.quicksum((product_value[j] * x[i, j] for i in platform_ids for j in product_names))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)