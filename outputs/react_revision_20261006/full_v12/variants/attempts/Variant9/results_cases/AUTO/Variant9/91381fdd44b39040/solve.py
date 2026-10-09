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
 'route': 'Others',
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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    item_frame = CSVQA_FRAMES['file_0_view_0']
    pattern_frame = CSVQA_FRAMES['file_1_view_0']
    items = []
    demand = {}
    for (_, row) in item_frame.iterrows():
        item = row['Item']
        items.append(item)
        try:
            demand[item] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for item {item}: {row['Demand']}")
    patterns = []
    for (_, row) in pattern_frame.iterrows():
        patterns.append(row['Pattern'])
    a = {}
    for (_, row) in pattern_frame.iterrows():
        p = row['Pattern']
        a[p] = {}
        for i in items:
            try:
                a[p][i] = float(row[i])
            except Exception:
                raise ValueError(f'Invalid coefficient for pattern {p}, item {i}: {row[i]}')
    for i in items:
        if i not in item_frame['Item'].values:
            raise ValueError(f'Item {i} missing from item demand data.')
    for p in patterns:
        if p not in pattern_frame['Pattern'].values:
            raise ValueError(f'Pattern {p} missing from pattern data.')
        for i in items:
            if i not in a[p]:
                raise ValueError(f'Pattern {p} missing coefficient for item {i}.')
    m = gp.Model('CuttingStockPatternSelection')
    y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
    for i in items:
        m.addConstr(gp.quicksum((a[p][i] * y_vars[p] for p in patterns)) >= demand[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)