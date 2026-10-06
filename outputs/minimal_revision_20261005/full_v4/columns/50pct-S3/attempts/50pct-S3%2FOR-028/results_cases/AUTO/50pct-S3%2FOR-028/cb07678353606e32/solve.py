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
 'relationships': [{'column_axis': {'id_column': 'Warehouse (i)', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'W1': '1',
                                          'W10': '10',
                                          'W11': '11',
                                          'W2': '2',
                                          'W3': '3',
                                          'W4': '4',
                                          'W5': '5',
                                          'W6': '6',
                                          'W7': '7',
                                          'W8': '8',
                                          'W9': '9'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Store (j)', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 1',
                    'row_id_mapping': {'W1': '1',
                                       'W10': '10',
                                       'W11': '11',
                                       'W2': '2',
                                       'W3': '3',
                                       'W4': '4',
                                       'W5': '5',
                                       'W6': '6',
                                       'W7': '7',
                                       'W8': '8',
                                       'W9': '9'},
                    'type': 'matrix'}],
 'route': 'FLP',
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
             'role': 'potential warehouse costs and capacities',
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
             'role': 'store demands',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 1', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11'],
             'file_index': 2,
             'file_name': 'TransportationCost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 1': 'W1',
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
                          'values': {'Unnamed: 1': 'W2',
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
                          'values': {'Unnamed: 1': 'W3',
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
                          'values': {'Unnamed: 1': 'W4',
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
                          'values': {'Unnamed: 1': 'W5',
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
                          'values': {'Unnamed: 1': 'W6',
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
                          'values': {'Unnamed: 1': 'W7',
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
                          'values': {'Unnamed: 1': 'W8',
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
                          'values': {'Unnamed: 1': 'W9',
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
                          'values': {'Unnamed: 1': 'W10',
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
                          'values': {'Unnamed: 1': 'W11',
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
             'role': 'warehouse-store transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [11, 11],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'unique_complete_suffix',
                                   'shape': [11, 11]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    warehouse_table_id = 'file_0_view_0'
    store_table_id = 'file_1_view_0'
    cost_matrix_table_id = 'file_2_view_0'
    warehouse_records = [r['values'] for r in next((t for t in data['tables'] if t['table_id'] == warehouse_table_id))['records']]
    I = [rec['Warehouse (i)'] for rec in warehouse_records]
    f = {rec['Warehouse (i)']: float(rec['Opening Cost (fi)']) for rec in warehouse_records}
    u = {rec['Warehouse (i)']: float(rec['Capacity (units)']) for rec in warehouse_records}
    store_records = [r['values'] for r in next((t for t in data['tables'] if t['table_id'] == store_table_id))['records']]
    J = [rec['Store (j)'] for rec in store_records]
    d = {rec['Store (j)']: float(rec['Demand (units, dj)']) for rec in store_records}
    rel = next((r for r in data['relationships'] if r['matrix_table_id'] == cost_matrix_table_id))
    col_id_map = rel.get('column_id_mapping', {})
    row_id_map = rel.get('row_id_mapping', {})
    matrix_table = next((t for t in data['tables'] if t['table_id'] == cost_matrix_table_id))
    matrix_records = [r['values'] for r in matrix_table['records']]
    c = {}
    for i in I:
        matrix_row_label = None
        for (k, v) in row_id_map.items():
            if v == i:
                matrix_row_label = k
                break
        if matrix_row_label is None:
            for rec in matrix_records:
                if rec['Unnamed: 1'] == i:
                    matrix_row_label = i
                    break
        if matrix_row_label is None:
            raise ValueError(f'Missing row mapping for warehouse {i}')
        row_rec = next((rec for rec in matrix_records if rec['Unnamed: 1'] == matrix_row_label), None)
        if row_rec is None:
            raise ValueError(f'Missing matrix row for warehouse {i} (label {matrix_row_label})')
        for j in J:
            matrix_col_label = None
            for (k, v) in col_id_map.items():
                if v == j:
                    matrix_col_label = k
                    break
            if matrix_col_label is None:
                if j in row_rec:
                    matrix_col_label = j
                else:
                    raise ValueError(f'Missing column mapping for store {j}')
            cij = row_rec.get(matrix_col_label)
            if cij is None:
                raise ValueError(f'Missing c_ij for warehouse {i} (row {matrix_row_label}), store {j} (col {matrix_col_label})')
            c.setdefault(i, {})[j] = float(cij)
    for i in I:
        if i not in f or i not in u or i not in c:
            raise ValueError(f'Missing data for warehouse {i}')
        for j in J:
            if j not in c[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('FLP')
    x_keys = [(i, j) for i in I for j in J]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == d[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= u[i] * y[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()