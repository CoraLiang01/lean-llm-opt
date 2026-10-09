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
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [11, 11],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'unique_complete_suffix',
                                   'shape': [11, 11]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    warehouses_df = CSVQA_FRAMES['file_0_view_0']
    stores_df = CSVQA_FRAMES['file_1_view_0']
    cost_matrix_df = CSVQA_FRAMES['file_2_view_0']
    I = []
    f_i = {}
    s_i = {}
    for (_, row) in warehouses_df.iterrows():
        i = row['Warehouse (i)']
        I.append(i)
        f_i[i] = float(row['Opening Cost (fi)'])
        s_i[i] = float(row['Capacity (units)'])
    J = []
    d_j = {}
    for (_, row) in stores_df.iterrows():
        j = row['Store (j)']
        J.append(j)
        d_j[j] = float(row['Demand (units, dj)'])
    col_mapping = None
    row_mapping = None
    for rel in CSVQA_DATA.get('relationships', []):
        if rel.get('matrix_table_id') == 'file_2_view_0':
            col_mapping = rel.get('column_id_mapping')
            row_mapping = rel.get('row_id_mapping')
            break
    store_id_to_row_label = {}
    if row_mapping:
        for (k, v) in row_mapping.items():
            store_id_to_row_label[v] = k
    else:
        for j in J:
            store_id_to_row_label[j] = j
    warehouse_id_to_col_label = {}
    if col_mapping:
        for (k, v) in col_mapping.items():
            warehouse_id_to_col_label[v] = k
    else:
        for i in I:
            warehouse_id_to_col_label[i] = i
    c_ij = {i: {} for i in I}
    for j in J:
        row_label = store_id_to_row_label[j]
        row = cost_matrix_df[cost_matrix_df['Unnamed: 3'] == row_label]
        if row.empty:
            raise ValueError(f'Missing transportation cost row for store {j} (row label {row_label})')
        row = row.iloc[0]
        for i in I:
            col_label = warehouse_id_to_col_label[i]
            if col_label not in cost_matrix_df.columns:
                raise ValueError(f'Missing transportation cost column for warehouse {i} (col label {col_label})')
            val = row[col_label]
            try:
                c_ij[i][j] = float(val)
            except Exception:
                raise ValueError(f'Invalid transportation cost for warehouse {i}, store {j}: {val}')
    for i in I:
        if i not in f_i or i not in s_i:
            raise ValueError(f'Missing opening cost or capacity for warehouse {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('FLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f_i[i] * y_vars[i] for i in I)) + gp.quicksum((c_ij[i][j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= s_i[i] for i in I), name='')
    m.addConstrs((x_vars[i, j] <= s_i[i] * y_vars[i] for i in I for j in J), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')