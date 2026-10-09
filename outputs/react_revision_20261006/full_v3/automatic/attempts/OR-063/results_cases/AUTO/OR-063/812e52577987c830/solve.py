CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these '
          'warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is '
          "recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The "
          'objective is to determine which warehouses should be activated so that the demand of all musicians and '
          'bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or '
          'band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the '
          'total supply obtained from different warehouses.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '553'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '17106'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '594'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '732'}}],
             'returned_rows': 7,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '98.70999999999999'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.73'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.95999999999999'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '98.16'}}],
             'returned_rows': 7,
             'role': 'warehouse fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'C1': '1506.22',
                                     'C2': '70.90000000000001',
                                     'C3': '8.44',
                                     'C4': '260.27',
                                     'C5': '197.47',
                                     'C6': '71.70999999999999',
                                     'C7': '61.19',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65',
                                     'C2': '1780.72',
                                     'C3': '567.4400000000001',
                                     'C4': '448.68',
                                     'C5': '29',
                                     'C6': '1484.91',
                                     'C7': '963.92',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66',
                                     'C2': '100.76',
                                     'C3': '64.68000000000001',
                                     'C4': '1324.53',
                                     'C5': '64.98999999999999',
                                     'C6': '134.88',
                                     'C7': '2102.83',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1254.78',
                                     'C2': '1115.63',
                                     'C3': '52.31',
                                     'C4': '1036.16',
                                     'C5': '892.63',
                                     'C6': '1464.04',
                                     'C7': '1383.41',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '42.9',
                                     'C2': '891.01',
                                     'C3': '1013.94',
                                     'C4': '1128.72',
                                     'C5': '58.91',
                                     'C6': '42.89',
                                     'C7': '1570.31',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '0.7',
                                     'C2': '139.46',
                                     'C3': '70.03',
                                     'C4': '79.15000000000001',
                                     'C5': '1482',
                                     'C6': '0.91',
                                     'C7': '110.46',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '1732.3',
                                     'C2': '1780.44',
                                     'C3': '486.5',
                                     'C4': '523.74',
                                     'C5': '522.08',
                                     'C6': '82.48',
                                     'C7': '826.41',
                                     'Unnamed: 0': 'S7'}}],
             'returned_rows': 7,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [7, 7],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [7, 7]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    warehouses = []
    warehouse_set = set()
    for rec in data['tables']:
        if rec['table_id'] == 'file_1_view_0':
            for r in rec['records']:
                wid = r['values']['Unnamed: 0']
                if wid not in warehouse_set:
                    warehouses.append(wid)
                    warehouse_set.add(wid)
        if rec['table_id'] == 'file_2_view_0':
            for r in rec['records']:
                wid = r['values']['Unnamed: 0']
                if wid not in warehouse_set:
                    warehouses.append(wid)
                    warehouse_set.add(wid)
    musicians = []
    musician_set = set()
    for rec in data['tables']:
        if rec['table_id'] == 'file_0_view_0':
            for r in rec['records']:
                cid = r['values']['customer']
                if cid not in musician_set:
                    musicians.append(cid)
                    musician_set.add(cid)
        if rec['table_id'] == 'file_2_view_0':
            for cname in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
                if cname not in musician_set:
                    musicians.append(cname)
                    musician_set.add(cname)
    demand = {}
    for rec in data['tables']:
        if rec['table_id'] == 'file_0_view_0':
            for r in rec['records']:
                cid = r['values']['customer']
                val = r['values']['demand']
                demand[cid] = float(val)
    fixed_cost = {}
    for rec in data['tables']:
        if rec['table_id'] == 'file_1_view_0':
            for r in rec['records']:
                wid = r['values']['Unnamed: 0']
                val = r['values']['fixed_costs']
                fixed_cost[wid] = float(val)
    cost = {}
    for rec in data['tables']:
        if rec['table_id'] == 'file_2_view_0':
            for r in rec['records']:
                wid = r['values']['Unnamed: 0']
                cost[wid] = {}
                for cname in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
                    val = r['values'][cname]
                    cost[wid][cname] = float(val)
    M = sum((demand[j] for j in musicians if j in demand))
    for i in warehouses:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for warehouse {i}')
        for j in musicians:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, musician {j}')
    for j in musicians:
        if j not in demand:
            raise ValueError(f'Missing demand for musician {j}')
    m = gp.Model('Bandcamp_FLP')
    x_keys = [(i, j) for i in warehouses for j in musicians]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in musicians)) + gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in musicians), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in musicians)) <= M * y[i] for i in warehouses), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()