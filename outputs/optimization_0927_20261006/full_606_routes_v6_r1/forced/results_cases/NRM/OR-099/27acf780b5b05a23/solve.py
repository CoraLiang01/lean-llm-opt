CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail company wants to decide where to open warehouses to minimize total costs while meeting the demands '
          'of several stores in a region. Each potential warehouse has an opening cost, and there is a known '
          'transportation cost for supplying goods to each store from each warehouse. Each warehouse has a capacity, '
          'and each store has a specific demand. The potential warehouses costs are in PotentialWarehouses_Costs.csv, '
          'demand for each store is in Stores_Demands.csv and the transportation costs c_ij from warehouse i to store '
          'j is in TransportationCost.csv\n'
          '\n'
          '    You are to determine which warehouses to open and how to assign the demand of each store to the '
          'warehouses such that all store demands are met, warehouse capacities are not exceeded, and the total cost '
          '(opening + transportation) is minimized.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Warehouse (i)', 'Opening Cost (fi)', 'Capacity (units)'],
             'file_index': 0,
             'file_name': 'PotentialWarehouses_Costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '3000', 'Warehouse (i)': '1'}},
                         {'source_row': 1,
                          'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '3200', 'Warehouse (i)': '2'}},
                         {'source_row': 2,
                          'values': {'Capacity (units)': '200', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '3'}},
                         {'source_row': 3,
                          'values': {'Capacity (units)': '150', 'Opening Cost (fi)': '2800', 'Warehouse (i)': '4'}},
                         {'source_row': 4,
                          'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3500', 'Warehouse (i)': '5'}},
                         {'source_row': 5,
                          'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2700', 'Warehouse (i)': '6'}},
                         {'source_row': 6,
                          'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '2900', 'Warehouse (i)': '7'}},
                         {'source_row': 7,
                          'values': {'Capacity (units)': '175', 'Opening Cost (fi)': '3050', 'Warehouse (i)': '8'}},
                         {'source_row': 8,
                          'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '9'}},
                         {'source_row': 9,
                          'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '2200', 'Warehouse (i)': '10'}},
                         {'source_row': 10,
                          'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2890', 'Warehouse (i)': '11'}}],
             'returned_rows': 11,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Store (j)', 'Demand (units, dj)'],
             'file_index': 1,
             'file_name': 'Stores_Demands.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0, 'values': {'Demand (units, dj)': '30', 'Store (j)': '1'}},
                         {'source_row': 1, 'values': {'Demand (units, dj)': '40', 'Store (j)': '2'}},
                         {'source_row': 2, 'values': {'Demand (units, dj)': '20', 'Store (j)': '3'}},
                         {'source_row': 3, 'values': {'Demand (units, dj)': '35', 'Store (j)': '4'}},
                         {'source_row': 4, 'values': {'Demand (units, dj)': '20', 'Store (j)': '5'}},
                         {'source_row': 5, 'values': {'Demand (units, dj)': '25', 'Store (j)': '6'}},
                         {'source_row': 6, 'values': {'Demand (units, dj)': '45', 'Store (j)': '7'}},
                         {'source_row': 7, 'values': {'Demand (units, dj)': '38', 'Store (j)': '8'}},
                         {'source_row': 8, 'values': {'Demand (units, dj)': '32', 'Store (j)': '9'}},
                         {'source_row': 9, 'values': {'Demand (units, dj)': '41', 'Store (j)': '10'}},
                         {'source_row': 10, 'values': {'Demand (units, dj)': '44', 'Store (j)': '11'}}],
             'returned_rows': 11,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11'],
             'file_index': 2,
             'file_name': 'TransportationCost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 0': 'W1',
                                     'W1': '12',
                                     'W10': '14',
                                     'W11': '15',
                                     'W2': '11',
                                     'W3': '14',
                                     'W4': '15',
                                     'W5': '17',
                                     'W6': '13',
                                     'W7': '12',
                                     'W8': '16',
                                     'W9': '16'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 0': 'W2',
                                     'W1': '17',
                                     'W10': '15',
                                     'W11': '16',
                                     'W2': '19',
                                     'W3': '15',
                                     'W4': '20',
                                     'W5': '18',
                                     'W6': '14',
                                     'W7': '17',
                                     'W8': '15',
                                     'W9': '13'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 0': 'W3',
                                     'W1': '13',
                                     'W10': '18',
                                     'W11': '17',
                                     'W2': '14',
                                     'W3': '12',
                                     'W4': '14',
                                     'W5': '16',
                                     'W6': '15',
                                     'W7': '11',
                                     'W8': '14',
                                     'W9': '16'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 0': 'W4',
                                     'W1': '18',
                                     'W10': '13',
                                     'W11': '18',
                                     'W2': '16',
                                     'W3': '17',
                                     'W4': '13',
                                     'W5': '18',
                                     'W6': '17',
                                     'W7': '14',
                                     'W8': '19',
                                     'W9': '16'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 0': 'W5',
                                     'W1': '10',
                                     'W10': '15',
                                     'W11': '17',
                                     'W2': '13',
                                     'W3': '12',
                                     'W4': '19',
                                     'W5': '15',
                                     'W6': '11',
                                     'W7': '12',
                                     'W8': '14',
                                     'W9': '12'}},
                         {'source_row': 5,
                          'values': {'Unnamed: 0': 'W6',
                                     'W1': '15',
                                     'W10': '18',
                                     'W11': '19',
                                     'W2': '12',
                                     'W3': '14',
                                     'W4': '16',
                                     'W5': '13',
                                     'W6': '17',
                                     'W7': '16',
                                     'W8': '16',
                                     'W9': '14'}},
                         {'source_row': 6,
                          'values': {'Unnamed: 0': 'W7',
                                     'W1': '14',
                                     'W10': '16',
                                     'W11': '14',
                                     'W2': '13',
                                     'W3': '15',
                                     'W4': '17',
                                     'W5': '12',
                                     'W6': '13',
                                     'W7': '14',
                                     'W8': '15',
                                     'W9': '12'}},
                         {'source_row': 7,
                          'values': {'Unnamed: 0': 'W8',
                                     'W1': '19',
                                     'W10': '15',
                                     'W11': '18',
                                     'W2': '16',
                                     'W3': '18',
                                     'W4': '20',
                                     'W5': '17',
                                     'W6': '19',
                                     'W7': '16',
                                     'W8': '18',
                                     'W9': '15'}},
                         {'source_row': 8,
                          'values': {'Unnamed: 0': 'W9',
                                     'W1': '17',
                                     'W10': '15',
                                     'W11': '18',
                                     'W2': '18',
                                     'W3': '12',
                                     'W4': '14',
                                     'W5': '16',
                                     'W6': '15',
                                     'W7': '14',
                                     'W8': '17',
                                     'W9': '21'}},
                         {'source_row': 9,
                          'values': {'Unnamed: 0': 'W10',
                                     'W1': '14',
                                     'W10': '17',
                                     'W11': '19',
                                     'W2': '13',
                                     'W3': '15',
                                     'W4': '17',
                                     'W5': '16',
                                     'W6': '18',
                                     'W7': '14',
                                     'W8': '19',
                                     'W9': '15'}},
                         {'source_row': 10,
                          'values': {'Unnamed: 0': 'W11',
                                     'W1': '15',
                                     'W10': '21',
                                     'W11': '13',
                                     'W2': '13',
                                     'W3': '16',
                                     'W4': '17',
                                     'W5': '11',
                                     'W6': '13',
                                     'W7': '14',
                                     'W8': '15',
                                     'W9': '19'}}],
             'returned_rows': 11,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [11, 11], "
                                   "'expected_shape': [11, 11], 'row_ids_aligned': False, 'column_ids_aligned': False, "
                                   "'row_order_matches': False, 'column_order_matches': False}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [11, 11], "
                                   "'expected_shape': [11, 11], 'row_ids_aligned': False, 'column_ids_aligned': False, "
                                   "'row_order_matches': False, 'column_order_matches': False}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
warehouses_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
warehouses_records = warehouses_table['records']
I = []
f = {}
C = {}
for rec in warehouses_records:
    i = rec['values']['Warehouse (i)']
    I.append(i)
    f[i] = float(rec['values']['Opening Cost (fi)'])
    C[i] = float(rec['values']['Capacity (units)'])
stores_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
stores_records = stores_table['records']
J = []
d = {}
for rec in stores_records:
    j = rec['values']['Store (j)']
    J.append(j)
    d[j] = float(rec['values']['Demand (units, dj)'])
trans_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
trans_records = trans_table['records']
warehouse_idx_to_W = {}
store_idx_to_W = {}
for (idx, i) in enumerate(I):
    warehouse_idx_to_W[i] = f'W{i}'
for (idx, j) in enumerate(J):
    store_idx_to_W[j] = f'W{j}'
c = {}
for rec in trans_records:
    i_W = rec['values']['Unnamed: 0']
    i = None
    for (key, val) in warehouse_idx_to_W.items():
        if val == i_W:
            i = key
            break
    if i is None:
        raise ValueError(f'Warehouse row {i_W} not found in warehouse index mapping.')
    for j in J:
        j_W = store_idx_to_W[j]
        if j_W not in rec['values']:
            raise ValueError(f'Store column {j_W} not found in transportation cost table.')
        c[i, j] = float(rec['values'][j_W])
if set(f.keys()) != set(I) or set(C.keys()) != set(I):
    raise ValueError('Mismatch in warehouse indices for opening cost or capacity.')
if set(d.keys()) != set(J):
    raise ValueError('Mismatch in store indices for demand.')
for i in I:
    for j in J:
        if (i, j) not in c:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}.')
m = gp.Model('Warehouse_Location')
y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
x_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((f[i] * y_vars[i] for i in I)) + gp.quicksum((c[i, j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == d[j] for j in J), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= C[i] * y_vars[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')