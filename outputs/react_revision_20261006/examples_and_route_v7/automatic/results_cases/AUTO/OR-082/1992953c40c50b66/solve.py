CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'A': '28',
                                     'B': '41',
                                     'C': '63',
                                     'D': '39',
                                     'Depot': '0',
                                     'E': '38',
                                     'F': '45',
                                     'G': '35',
                                     'H': '28',
                                     'I': '44',
                                     'J': '35',
                                     'Unnamed: 0': 'Depot'}},
                         {'source_row': 1,
                          'values': {'A': '0',
                                     'B': '27',
                                     'C': '87',
                                     'D': '35',
                                     'Depot': '28',
                                     'E': '65',
                                     'F': '63',
                                     'G': '41',
                                     'H': '39',
                                     'I': '43',
                                     'J': '20',
                                     'Unnamed: 0': 'A'}},
                         {'source_row': 2,
                          'values': {'A': '27',
                                     'B': '0',
                                     'C': '81',
                                     'D': '13',
                                     'Depot': '41',
                                     'E': '77',
                                     'F': '54',
                                     'G': '25',
                                     'H': '63',
                                     'I': '70',
                                     'J': '7',
                                     'Unnamed: 0': 'B'}},
                         {'source_row': 3,
                          'values': {'A': '87',
                                     'B': '81',
                                     'C': '0',
                                     'D': '69',
                                     'Depot': '63',
                                     'E': '53',
                                     'F': '28',
                                     'G': '57',
                                     'H': '83',
                                     'I': '102',
                                     'J': '81',
                                     'Unnamed: 0': 'C'}},
                         {'source_row': 4,
                          'values': {'A': '35',
                                     'B': '13',
                                     'C': '69',
                                     'D': '0',
                                     'Depot': '39',
                                     'E': '72',
                                     'F': '41',
                                     'G': '12',
                                     'H': '64',
                                     'I': '75',
                                     'J': '17',
                                     'Unnamed: 0': 'D'}},
                         {'source_row': 5,
                          'values': {'A': '65',
                                     'B': '77',
                                     'C': '53',
                                     'D': '72',
                                     'Depot': '38',
                                     'E': '0',
                                     'F': '53',
                                     'G': '64',
                                     'H': '39',
                                     'I': '58',
                                     'J': '72',
                                     'Unnamed: 0': 'E'}},
                         {'source_row': 6,
                          'values': {'A': '63',
                                     'B': '54',
                                     'C': '28',
                                     'D': '41',
                                     'Depot': '45',
                                     'E': '53',
                                     'F': '0',
                                     'G': '29',
                                     'H': '70',
                                     'I': '88',
                                     'J': '54',
                                     'Unnamed: 0': 'F'}},
                         {'source_row': 7,
                          'values': {'A': '41',
                                     'B': '25',
                                     'C': '57',
                                     'D': '12',
                                     'Depot': '35',
                                     'E': '64',
                                     'F': '29',
                                     'G': '0',
                                     'H': '63',
                                     'I': '76',
                                     'J': '27',
                                     'Unnamed: 0': 'G'}},
                         {'source_row': 8,
                          'values': {'A': '39',
                                     'B': '63',
                                     'C': '83',
                                     'D': '64',
                                     'Depot': '28',
                                     'E': '39',
                                     'F': '70',
                                     'G': '63',
                                     'H': '0',
                                     'I': '20',
                                     'J': '56',
                                     'Unnamed: 0': 'H'}},
                         {'source_row': 9,
                          'values': {'A': '43',
                                     'B': '70',
                                     'C': '102',
                                     'D': '75',
                                     'Depot': '44',
                                     'E': '58',
                                     'F': '88',
                                     'G': '76',
                                     'H': '20',
                                     'I': '0',
                                     'J': '63',
                                     'Unnamed: 0': 'I'}},
                         {'source_row': 10,
                          'values': {'A': '20',
                                     'B': '7',
                                     'C': '81',
                                     'D': '17',
                                     'Depot': '35',
                                     'E': '72',
                                     'F': '54',
                                     'G': '27',
                                     'H': '56',
                                     'I': '63',
                                     'J': '0',
                                     'Unnamed: 0': 'J'}}],
             'returned_rows': 11,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [4, 4], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [4, 4], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem():
    table_id = 'file_0_view_0'
    df = CSVQA_FRAMES[table_id]
    N = ['Depot', 'A', 'B', 'C']
    d = {}
    df_rows = df['Unnamed: 0'].tolist()
    for i in N:
        if i not in df_rows:
            raise ValueError(f'Missing row for node {i} in distance matrix.')
    for i in N:
        row = df[df['Unnamed: 0'] == i].iloc[0]
        for j in N:
            if j not in df.columns:
                raise ValueError(f'Missing column for node {j} in distance matrix.')
            val = row[j]
            try:
                d[i, j] = float(val)
            except Exception:
                raise ValueError(f'Invalid distance value for ({i},{j}): {val}')
    for i in N:
        for j in N:
            if (i, j) not in d:
                raise ValueError(f'Missing distance for ({i},{j})')
    m = gp.Model('TSP4')
    x_keys = [(i, j) for i in N for j in N if i != j]
    x_vars = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    customers = ['A', 'B', 'C']
    u_vars = m.addVars(customers, lb=1, ub=3, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((d[i, j] * x_vars[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
    for i in N:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'depart_{i}')
    for j in N:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'arrive_{j}')
    for i in customers:
        for j in customers:
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'subtour_{i}_{j}')
    for i in N:
        if (i, i) in x_vars:
            m.addConstr(x_vars[i, i] == 0, name=f'noloop_{i}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()