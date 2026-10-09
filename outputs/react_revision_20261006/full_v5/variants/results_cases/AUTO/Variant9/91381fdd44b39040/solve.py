CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A paper mill cuts standard rolls into smaller item types. The demand for each item type is provided in '
          'item_demand.csv. A set of feasible cutting patterns is provided in cutting_patterns.csv, where each pattern '
          'specifies how many units of each item type are produced by cutting one standard roll according to that '
          'pattern.\n'
          '\n'
          'Formulate an integer cutting-stock pattern-selection model. For each cutting pattern p, define y_p as the '
          'integer number of standard rolls cut using pattern p. The objective is to minimize the total number of '
          'standard rolls used. The model should satisfy or exceed demand for every item type, impose nonnegativity on '
          'pattern-use variables, and require pattern-use variables to be integers.',
 'relationships': [{'column_axis': {'id_column': 'Item', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_1_view_0',
                    'row_axis': {'id_column': 'Pattern', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Pattern',
                    'type': 'matrix'}],
 'route': 'RA',
 'tables': [{'columns': ['Item', 'Demand'],
             'file_index': 0,
             'file_name': 'item_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Demand': '24', 'Item': 'A'}},
                         {'source_row': 1, 'values': {'Demand': '18', 'Item': 'B'}},
                         {'source_row': 2, 'values': {'Demand': '12', 'Item': 'C'}},
                         {'source_row': 3, 'values': {'Demand': '10', 'Item': 'D'}}],
             'returned_rows': 4,
             'role': 'item demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Pattern', 'A', 'B', 'C', 'D'],
             'file_index': 1,
             'file_name': 'cutting_patterns.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0, 'values': {'A': '4', 'B': '0', 'C': '0', 'D': '0', 'Pattern': 'P1'}},
                         {'source_row': 1, 'values': {'A': '0', 'B': '3', 'C': '0', 'D': '0', 'Pattern': 'P2'}},
                         {'source_row': 2, 'values': {'A': '0', 'B': '0', 'C': '2', 'D': '0', 'Pattern': 'P3'}},
                         {'source_row': 3, 'values': {'A': '0', 'B': '0', 'C': '0', 'D': '2', 'Pattern': 'P4'}},
                         {'source_row': 4, 'values': {'A': '2', 'B': '1', 'C': '0', 'D': '0', 'Pattern': 'P5'}},
                         {'source_row': 5, 'values': {'A': '1', 'B': '0', 'C': '1', 'D': '0', 'Pattern': 'P6'}},
                         {'source_row': 6, 'values': {'A': '0', 'B': '1', 'C': '0', 'D': '1', 'Pattern': 'P7'}},
                         {'source_row': 7, 'values': {'A': '1', 'B': '1', 'C': '1', 'D': '0', 'Pattern': 'P8'}},
                         {'source_row': 8, 'values': {'A': '2', 'B': '0', 'C': '0', 'D': '1', 'Pattern': 'P9'}}],
             'returned_rows': 9,
             'role': 'cutting patterns',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [9, 4],
                                   'matrix_table_id': 'file_1_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [9, 4]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    items = [rec['values']['Item'] for rec in data['tables'][0]['records']]
    patterns = [rec['values']['Pattern'] for rec in data['tables'][1]['records']]
    demand = {rec['values']['Item']: int(rec['values']['Demand']) for rec in data['tables'][0]['records']}
    a = {}
    for rec in data['tables'][1]['records']:
        p = rec['values']['Pattern']
        for i in items:
            a[p, i] = int(rec['values'][i])
    m = gp.Model('cutting_stock_pattern_selection')
    y = m.addVars(patterns, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((y[p] for p in patterns)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((a[p, i] * y[p] for p in patterns)) >= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()