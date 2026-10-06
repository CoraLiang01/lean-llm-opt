CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A roll-cutting line has a fixed menu of feasible cutting patterns. Item demands are listed in '
          'item_demand.csv, and the number of each item produced by every pattern is listed in cutting_patterns.csv. A '
          'pattern may be used any nonnegative integer number of times, and producing extra pieces is allowed.\n'
          '\n'
          'Formulate a minimum-roll cutting-stock model. For each cutting pattern p, define y_p as the nonnegative '
          'integer number of stock rolls cut with pattern p. The objective is to minimize the total number of rolls '
          'used. The model should include demand satisfaction constraints for every item, nonnegativity constraints, '
          'and integer restrictions for all pattern-use variables.',
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
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Demand': '18', 'Item': 'A'}},
                         {'source_row': 1, 'values': {'Demand': '14', 'Item': 'B'}},
                         {'source_row': 2, 'values': {'Demand': '12', 'Item': 'C'}},
                         {'source_row': 3, 'values': {'Demand': '10', 'Item': 'D'}},
                         {'source_row': 4, 'values': {'Demand': '8', 'Item': 'E'}}],
             'returned_rows': 5,
             'role': 'item demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Pattern', 'A', 'B', 'C', 'D', 'E'],
             'file_index': 1,
             'file_name': 'cutting_patterns.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'A': '3', 'B': '0', 'C': '0', 'D': '0', 'E': '0', 'Pattern': 'P1'}},
                         {'source_row': 1,
                          'values': {'A': '0', 'B': '2', 'C': '1', 'D': '0', 'E': '0', 'Pattern': 'P2'}},
                         {'source_row': 2,
                          'values': {'A': '0', 'B': '0', 'C': '2', 'D': '1', 'E': '0', 'Pattern': 'P3'}},
                         {'source_row': 3,
                          'values': {'A': '0', 'B': '0', 'C': '0', 'D': '2', 'E': '1', 'Pattern': 'P4'}},
                         {'source_row': 4,
                          'values': {'A': '1', 'B': '1', 'C': '0', 'D': '1', 'E': '0', 'Pattern': 'P5'}},
                         {'source_row': 5,
                          'values': {'A': '2', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'Pattern': 'P6'}},
                         {'source_row': 6,
                          'values': {'A': '0', 'B': '1', 'C': '1', 'D': '0', 'E': '1', 'Pattern': 'P7'}},
                         {'source_row': 7,
                          'values': {'A': '1', 'B': '0', 'C': '0', 'D': '1', 'E': '1', 'Pattern': 'P8'}},
                         {'source_row': 8,
                          'values': {'A': '1', 'B': '2', 'C': '0', 'D': '0', 'E': '0', 'Pattern': 'P9'}},
                         {'source_row': 9,
                          'values': {'A': '0', 'B': '0', 'C': '1', 'D': '1', 'E': '1', 'Pattern': 'P10'}}],
             'returned_rows': 10,
             'role': 'cutting pattern matrix',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [10, 5],
                                   'matrix_table_id': 'file_1_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [10, 5]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    items = [rec['values']['Item'] for rec in data['tables'][0]['records']]
    patterns = [rec['values']['Pattern'] for rec in data['tables'][1]['records']]
    demand = {}
    for rec in data['tables'][0]['records']:
        item = rec['values']['Item']
        demand[item] = int(rec['values']['Demand'])
    a = {}
    for rec in data['tables'][1]['records']:
        p = rec['values']['Pattern']
        for i in items:
            a[p, i] = int(rec['values'][i])
    for i in items:
        if i not in demand:
            raise ValueError(f'Missing demand for item {i}')
    for p in patterns:
        for i in items:
            if (p, i) not in a:
                raise ValueError(f'Missing pattern coefficient for pattern {p}, item {i}')
    m = gp.Model('cutting_stock')
    y = m.addVars(patterns, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((y[p] for p in patterns)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((a[p, i] * y[p] for p in patterns)) >= demand[i] for i in items), name='')
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