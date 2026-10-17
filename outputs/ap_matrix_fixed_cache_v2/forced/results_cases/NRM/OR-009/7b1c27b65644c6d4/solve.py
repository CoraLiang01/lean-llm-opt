CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” '
          'while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost '
          'per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is '
          'to determine the optimal quantity of beverages to be shipped from each production plant to each retail '
          'outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_column_id_map': {},
                    'matrix_format': 'wide',
                    'matrix_metadata_columns': [],
                    'matrix_table_id': 'file_2_view_0',
                    'matrix_value_columns': ['C1', 'C2', 'C3', 'C4'],
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
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
             'role': 'supply capacity',
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
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_order_matches': True,
                                   'column_permutation': [0, 1, 2, 3],
                                   'expected_shape': [4, 4],
                                   'format': 'wide',
                                   'matrix_table_id': 'file_2_view_0',
                                   'metadata_columns': [],
                                   'row_ids_aligned': True,
                                   'row_order_matches': True,
                                   'row_permutation': [0, 1, 2, 3],
                                   'shape': [4, 4],
                                   'value_columns': ['C1', 'C2', 'C3', 'C4']}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']
customer_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
customers = [rec['values']['customer'] for rec in customer_table['records']]
demand = {rec['values']['customer']: float(rec['values']['demand']) for rec in customer_table['records']}
plant_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
plants = [rec['values']['Unnamed: 0'] for rec in plant_table['records']]
supply_capacity = {rec['values']['Unnamed: 0']: float(rec['values']['supply_capacity']) for rec in plant_table['records']}
cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
costs = {}
for rec in cost_table['records']:
    s = rec['values']['Unnamed: 0']
    costs[s] = {}
    for c in customers:
        if c not in rec['values']:
            raise ValueError(f'Missing transportation cost for plant {s} to customer {c}')
        costs[s][c] = float(rec['values'][c])
if set(customers) != set(cost_table['columns'][1:]):
    raise ValueError('Customer columns in transportation_costs.csv do not match customer_demand.csv')
if set(plants) != set([rec['values']['Unnamed: 0'] for rec in cost_table['records']]):
    raise ValueError('Plant rows in transportation_costs.csv do not match supply_capacity.csv')
m = gp.Model('BrewCo_Distribution')
x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((costs[s][c] * x_vars[s, c] for s in plants for c in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[s, c] for s in plants)) == demand[c] for c in customers), name='')
m.addConstrs((gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s] for s in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')