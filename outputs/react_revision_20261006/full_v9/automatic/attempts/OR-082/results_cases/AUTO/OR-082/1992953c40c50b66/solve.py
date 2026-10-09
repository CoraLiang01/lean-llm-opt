CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}},
                         {'source_row': 1,
                          'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}},
                         {'source_row': 2,
                          'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}},
                         {'source_row': 3,
                          'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}},
                         {'source_row': 4,
                          'values': {'A': '35', 'B': '13', 'C': '69', 'Depot': '39', 'Unnamed: 0': 'D'}},
                         {'source_row': 5,
                          'values': {'A': '65', 'B': '77', 'C': '53', 'Depot': '38', 'Unnamed: 0': 'E'}},
                         {'source_row': 6,
                          'values': {'A': '63', 'B': '54', 'C': '28', 'Depot': '45', 'Unnamed: 0': 'F'}},
                         {'source_row': 7,
                          'values': {'A': '41', 'B': '25', 'C': '57', 'Depot': '35', 'Unnamed: 0': 'G'}},
                         {'source_row': 8,
                          'values': {'A': '39', 'B': '63', 'C': '83', 'Depot': '28', 'Unnamed: 0': 'H'}},
                         {'source_row': 9,
                          'values': {'A': '43', 'B': '70', 'C': '102', 'Depot': '44', 'Unnamed: 0': 'I'}},
                         {'source_row': 10,
                          'values': {'A': '20', 'B': '7', 'C': '81', 'Depot': '35', 'Unnamed: 0': 'J'}}],
             'returned_rows': 11,
             'role': 'distance matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import sys

def solve_problem(CSVQA_FRAMES):
    table_id = 'file_0_view_0'
    frame = CSVQA_FRAMES[table_id]
    nodes = ['Depot', 'A', 'B', 'C']
    customers = ['A', 'B', 'C']
    d = {}
    for (source_row, row) in frame.iterrows():
        i = row['Unnamed: 0']
        if i not in nodes:
            continue
        for j in nodes:
            if i == j:
                continue
            val = row[j]
            try:
                d[i, j] = float(val)
            except Exception:
                raise ValueError(f'Distance from {i} to {j} is missing or not numeric: {val}')
    for i in nodes:
        for j in nodes:
            if i != j and (i, j) not in d:
                raise ValueError(f'Missing distance for ({i},{j})')
    m = gp.Model('TSP4')
    x_keys = [(i, j) for i in nodes for j in nodes if i != j]
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = m.addVars(customers, lb=1, ub=3, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((d[i, j] * x_vars[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in nodes if i != j)) == 1, name=f'enter_{j}')
    for i in customers:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in nodes if j != i)) == 1, name=f'leave_{i}')
    m.addConstr(gp.quicksum((x_vars['Depot', j] for j in nodes if j != 'Depot')) == 1, name='leave_depot')
    m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in nodes if i != 'Depot')) == 1, name='enter_depot')
    for i in customers:
        for j in customers:
            if i == j:
                continue
            m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'mtz_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)