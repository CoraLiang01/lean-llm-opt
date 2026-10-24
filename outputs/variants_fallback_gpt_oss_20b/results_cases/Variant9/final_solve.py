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
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Item', 'Demand'],
             'file_index': 0,
             'file_name': 'item_demand.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Demand': '24', 'Item': 'A'}},
                         {'source_row': 1, 'values': {'Demand': '18', 'Item': 'B'}},
                         {'source_row': 2, 'values': {'Demand': '12', 'Item': 'C'}},
                         {'source_row': 3, 'values': {'Demand': '10', 'Item': 'D'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Pattern', 'A', 'B', 'C', 'D'],
             'file_index': 1,
             'file_name': 'cutting_patterns.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    tables = CSVQA_DATA['tables']
    item_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    demand = {}
    for rec in item_table['records']:
        item = rec['values']['Item']
        dem = float(rec['values']['Demand'])
        demand[item] = dem
    pattern_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    patterns = {}
    for rec in pattern_table['records']:
        pat = rec['values']['Pattern']
        coeffs = {}
        for col in pattern_table['columns']:
            if col == 'Pattern':
                continue
            coeffs[col] = float(rec['values'][col])
        patterns[pat] = coeffs
    items = set(demand.keys())
    for (pat, coeffs) in patterns.items():
        if set(coeffs.keys()) != items:
            raise ValueError(f'Pattern {pat} does not provide all items')
    y = model.addVars(patterns.keys(), vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((y[p] for p in patterns.keys())), GRB.MINIMIZE)
    for i in items:
        model.addConstr(gurobipy.quicksum((patterns[p][i] * y[p] for p in patterns.keys())) >= demand[i], name=f'dem_{i}')
    model.optimize()
    return model
m = solve_problem()