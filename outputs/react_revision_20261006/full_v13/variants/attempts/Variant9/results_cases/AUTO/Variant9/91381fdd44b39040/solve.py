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
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import sys

def solve_problem(CSVQA_FRAMES):
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    patterns_frame = CSVQA_FRAMES['file_1_view_0']
    items = list(demand_frame['Item'])
    patterns = list(patterns_frame['Pattern'])
    d = {}
    for (idx, row) in demand_frame.iterrows():
        item = row['Item']
        try:
            d[item] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for item {item}: {row['Demand']}")
    a = {item: {} for item in items}
    for (idx, row) in patterns_frame.iterrows():
        pattern = row['Pattern']
        for item in items:
            try:
                a[item][pattern] = float(row[item])
            except Exception:
                raise ValueError(f'Non-numeric or missing a_{{ip}} for item {item}, pattern {pattern}: {row[item]}')
    m = gp.Model('CuttingStockPatternSelection')
    y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
    for item in items:
        m.addConstr(gp.quicksum((a[item][p] * y_vars[p] for p in patterns)) >= d[item], name=f'demand_{item}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'Optimal objective value: {m.objVal:.4f}')
        for p in patterns:
            print(f'y_{p}: {y_vars[p].VarName} = {y_vars[p].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem(CSVQA_FRAMES)