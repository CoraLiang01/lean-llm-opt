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
                    'row_id_column': 'Unnamed: 3',
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
            {'columns': ['Unnamed: 3', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11'],
             'file_index': 2,
             'file_name': 'TransportationCost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 3': 'W1',
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
                          'values': {'Unnamed: 3': 'W2',
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
                          'values': {'Unnamed: 3': 'W3',
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
                          'values': {'Unnamed: 3': 'W4',
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
                          'values': {'Unnamed: 3': 'W5',
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
                          'values': {'Unnamed: 3': 'W6',
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
                          'values': {'Unnamed: 3': 'W7',
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
                          'values': {'Unnamed: 3': 'W8',
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
                          'values': {'Unnamed: 3': 'W9',
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
                          'values': {'Unnamed: 3': 'W10',
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
                          'values': {'Unnamed: 3': 'W11',
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
    cost_table_id = 'file_2_view_0'
    warehouse_records = [r['values'] for r in next((t for t in data['tables'] if t['table_id'] == warehouse_table_id))['records']]
    I = [rec['Warehouse (i)'] for rec in warehouse_records]
    f = {rec['Warehouse (i)']: float(rec['Opening Cost (fi)']) for rec in warehouse_records}
    u = {rec['Warehouse (i)']: float(rec['Capacity (units)']) for rec in warehouse_records}
    store_records = [r['values'] for r in next((t for t in data['tables'] if t['table_id'] == store_table_id))['records']]
    J = [rec['Store (j)'] for rec in store_records]
    d = {rec['Store (j)']: float(rec['Demand (units, dj)']) for rec in store_records}
    rel = next((r for r in data['relationships'] if r['matrix_table_id'] == cost_table_id))
    row_id_mapping = rel.get('row_id_mapping', {})
    col_id_mapping = rel.get('column_id_mapping', {})
    row_id_column = rel['row_id_column']
    warehouse_to_rowlabel = {i: None for i in I}
    for (label, i) in row_id_mapping.items():
        warehouse_to_rowlabel[i] = label
    if not any(warehouse_to_rowlabel.values()):
        warehouse_to_rowlabel = {i: i for i in I}
    else:
        for i in I:
            if warehouse_to_rowlabel[i] is None:
                warehouse_to_rowlabel[i] = i
    store_to_collabel = {j: None for j in J}
    for (label, j) in col_id_mapping.items():
        store_to_collabel[j] = label
    if not any(store_to_collabel.values()):
        store_to_collabel = {j: j for j in J}
    else:
        for j in J:
            if store_to_collabel[j] is None:
                store_to_collabel[j] = j
    cost_records = [r['values'] for r in next((t for t in data['tables'] if t['table_id'] == cost_table_id))['records']]
    rowlabel_to_record = {rec[row_id_column]: rec for rec in cost_records}
    c = {}
    for i in I:
        rowlabel = warehouse_to_rowlabel[i]
        if rowlabel not in rowlabel_to_record:
            raise ValueError(f'Missing row for warehouse {i} (row label {rowlabel}) in cost matrix')
        row = rowlabel_to_record[rowlabel]
        for j in J:
            collabel = store_to_collabel[j]
            if collabel not in row:
                raise ValueError(f'Missing column for store {j} (column label {collabel}) in cost matrix')
            try:
                c_ij = float(row[collabel])
            except Exception:
                raise ValueError(f'Non-numeric cost for warehouse {i}, store {j}: {row[collabel]}')
            c[i, j] = c_ij
    for i in I:
        if i not in f or i not in u:
            raise ValueError(f'Missing opening cost or capacity for warehouse {i}')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for store {j}')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
    m = gp.Model('FLP')
    x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
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