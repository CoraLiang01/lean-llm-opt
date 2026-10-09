CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A roll-cutting line has a fixed menu of feasible cutting patterns. Item demands are listed in '
          'item_demand.csv, and the number of each item produced by every pattern is listed in cutting_patterns.csv. A '
          'pattern may be used any nonnegative integer number of times, and producing extra pieces is allowed.\n'
          '\n'
          'Formulate a minimum-roll cutting-stock model. For each cutting pattern p, define y_p as the nonnegative '
          'integer number of stock rolls cut with pattern p. The objective is to minimize the total number of rolls '
          'used. The model should include demand satisfaction constraints for every item, nonnegativity constraints, '
          'and integer restrictions for all pattern-use variables.',
 'relationships': [],
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
             'role': 'cutting patterns',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    item_frame = CSVQA_FRAMES['file_0_view_0']
    pattern_frame = CSVQA_FRAMES['file_1_view_0']
    items = []
    demand = {}
    for (_, row) in item_frame.iterrows():
        item = row['Item']
        items.append(item)
        demand[item] = float(row['Demand'])
    patterns = []
    for (_, row) in pattern_frame.iterrows():
        patterns.append(row['Pattern'])
    a_ip = {}
    for (_, row) in pattern_frame.iterrows():
        p = row['Pattern']
        for i in items:
            a_ip[i, p] = float(row[i])
    m = gp.Model('cutting_stock_min_rolls')
    y_vars = m.addVars(patterns, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((a_ip[i, p] * y_vars[p] for p in patterns)) >= demand[i] for i in items), name='')
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