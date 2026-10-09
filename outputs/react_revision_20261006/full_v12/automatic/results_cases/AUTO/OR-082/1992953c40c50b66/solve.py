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
    nodes = list(frame['Unnamed: 0'])
    columns = [col for col in frame.columns if col != 'Unnamed: 0']
    if set(nodes) != set(columns):
        raise ValueError('Row and column node sets do not match in distance matrix.')
    N = nodes
    V = [n for n in N if n != 'Depot']
    d = {}
    for (idx, row) in frame.iterrows():
        i = row['Unnamed: 0']
        d[i] = {}
        for j in columns:
            val = row[j]
            try:
                d[i][j] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric or missing distance from {i} to {j}: {val}')
    for i in N:
        for j in N:
            if j not in d[i]:
                raise ValueError(f'Missing distance from {i} to {j}')
    x_keys = [(i, j) for i in N for j in N if i != j]
    u_keys = V
    m = gp.Model('TSP')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = m.addVars(u_keys, lb=1, ub=len(V), vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((d[i][j] * x_vars[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in N:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'leave_{i}')
    for j in N:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'enter_{j}')
    for i in V:
        for j in V:
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + len(V) * x_vars[i, j] <= len(V) - 1, name=f'mtz_{i}_{j}')
    for i in V:
        m.addConstr(u_vars[i] >= 1, name=f'u_lb_{i}')
        m.addConstr(u_vars[i] <= len(V), name=f'u_ub_{i}')
    for i in N:
        if (i, i) in x_vars:
            m.addConstr(x_vars[i, i] == 0, name=f'no_self_{i}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Status {m.Status}')