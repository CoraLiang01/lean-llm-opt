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
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'visit three customer locations — A, B, and C — exactly once each '
                                                     'in any order, and then return to the depot',
                                         'inclusive': 'both',
                                         'operator': 'in',
                                         'value': ['Depot', 'A', 'B', 'C']}],
                         'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}},
                         {'source_row': 1,
                          'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}},
                         {'source_row': 2,
                          'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}},
                         {'source_row': 3,
                          'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}],
             'returned_rows': 4,
             'role': 'distance matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    N = []
    V = []
    for (idx, row) in frame.iterrows():
        loc = row['Unnamed: 0']
        N.append(loc)
    V = [loc for loc in N if loc != 'Depot']
    d = {}
    for (idx, row) in frame.iterrows():
        i = row['Unnamed: 0']
        for j in ['Depot', 'A', 'B', 'C']:
            if i != j:
                try:
                    d_ij = float(row[j])
                except Exception:
                    raise ValueError(f'Distance from {i} to {j} is not a valid number: {row[j]}')
                d[i, j] = d_ij
    for i in N:
        for j in N:
            if i != j and (i, j) not in d:
                raise ValueError(f'Missing distance from {i} to {j}')
    m = gp.Model('TSP_MTZ')
    x_keys = [(i, j) for i in N for j in N if i != j]
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = m.addVars(V, lb=1, ub=3, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((d[i, j] * x_vars[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for j in V:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1)
    for i in V:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1)
    m.addConstr(gp.quicksum((x_vars['Depot', j] for j in V)) == 1)
    m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in V)) == 1)
    for i in V:
        for j in V:
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)