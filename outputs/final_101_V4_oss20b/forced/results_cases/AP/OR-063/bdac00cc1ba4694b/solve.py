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
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [7, 7],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [7, 7]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
import numpy as np
demand = {}
for rec in CSVQA_DATA['tables'][0]['records']:
    customer = rec['values']['customer']
    demand[customer] = float(rec['values']['demand'])
fixed_costs = {}
for rec in CSVQA_DATA['tables'][1]['records']:
    warehouse = rec['values']['Unnamed: 0']
    fixed_costs[warehouse] = float(rec['values']['fixed_costs'])
transport = {}
for rec in CSVQA_DATA['tables'][2]['records']:
    warehouse = rec['values']['Unnamed: 0']
    transport[warehouse] = {}
    for cust in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        transport[warehouse][cust] = float(rec['values'][cust])
warehouses = [rec['values']['Unnamed: 0'] for rec in CSVQA_DATA['tables'][1]['records']]
customers = [rec['values']['customer'] for rec in CSVQA_DATA['tables'][0]['records']]
f = np.array([fixed_costs[w] for w in warehouses])
d = np.array([demand[c] for c in customers])
c = np.zeros((len(warehouses), len(customers)))
for i, w in enumerate(warehouses):
    for j, cust in enumerate(customers):
        c[i, j] = transport[w][cust]
m = gp.Model('Warehouse_Allocation')
y = m.addVars(len(warehouses), vtype=GRB.BINARY)
x = {}
for i, w in enumerate(warehouses):
    for j, cust in enumerate(customers):
        x[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
obj = gp.quicksum((f[i] * y[i] for i in range(len(warehouses)))) + gp.quicksum((c[i, j] * x[i, j] for i in range(len(warehouses)) for j in range(len(customers))))
m.setObjective(obj, GRB.MINIMIZE)
for j, cust in enumerate(customers):
    m.addConstr(gp.quicksum((x[i, j] for i in range(len(warehouses)))) == d[j])
for i, w in enumerate(warehouses):
    for j, cust in enumerate(customers):
        m.addConstr(x[i, j] <= d[j] * y[i])
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)