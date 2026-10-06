CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” '
          'while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost '
          'per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is '
          'to determine the optimal quantity of beverages to be shipped from each production plant to each retail '
          'outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '94'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '39'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '65'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'plant supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'C1': '543.756480860856',
                                     'C2': '23.685276141764653',
                                     'C3': '23.676386730773032',
                                     'C4': '447.75143678673766',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '883.9151090405642',
                                     'C2': '0.04977684765576961',
                                     'C3': '0.0350986687216299',
                                     'C4': '44.45588531711622',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '537.3456896658107',
                                     'C2': '23.769274659075112',
                                     'C3': '498.95659249465467',
                                     'C4': '440.60737890439776',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1791.493192397229',
                                     'C2': '68.21633865655126',
                                     'C3': '1432.4837339656747',
                                     'C4': '1527.7635425462734',
                                     'Unnamed: 0': 'S4'}}],
             'returned_rows': 4,
             'role': 'plant-to-customer transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [4, 4]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    plants = [rec['values']['Unnamed: 0'] for rec in CSVQA_DATA['tables'][1]['records']]
    outlets = [rec['values']['customer'] for rec in CSVQA_DATA['tables'][0]['records']]
    demand = {}
    for rec in CSVQA_DATA['tables'][0]['records']:
        j = rec['values']['customer']
        try:
            demand[j] = float(rec['values']['demand'])
        except Exception:
            raise ValueError(f'Invalid demand value for outlet {j}')
    supply = {}
    for rec in CSVQA_DATA['tables'][1]['records']:
        i = rec['values']['Unnamed: 0']
        try:
            supply[i] = float(rec['values']['supply_capacity'])
        except Exception:
            raise ValueError(f'Invalid supply value for plant {i}')
    cost = {}
    for rec in CSVQA_DATA['tables'][2]['records']:
        i = rec['values']['Unnamed: 0']
        cost[i] = {}
        for j in outlets:
            try:
                cost[i][j] = float(rec['values'][j])
            except Exception:
                raise ValueError(f'Missing or invalid cost for ({i},{j})')
    if set(plants) != set(cost.keys()):
        raise ValueError('Mismatch between plants and cost matrix rows')
    for i in plants:
        if set(outlets) != set(cost[i].keys()):
            raise ValueError(f'Mismatch between outlets and cost matrix columns for plant {i}')
    if set(plants) != set(supply.keys()):
        raise ValueError('Mismatch between plants and supply data')
    if set(outlets) != set(demand.keys()):
        raise ValueError('Mismatch between outlets and demand data')
    m = gp.Model('BrewCo_TP')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in plants for j in outlets]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for (i, j) in keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) >= demand[j] for j in outlets), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in outlets)) <= supply[i] for i in plants), name='')
    m.optimize()
    return m
m = solve_problem()