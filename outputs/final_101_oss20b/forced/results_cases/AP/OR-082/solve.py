CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [{'column_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_0_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    columns = table['columns']
    node_ids = [rec['values']['Unnamed: 0'] for rec in records]
    col_ids = [c for c in columns if c != 'Unnamed: 0']
    if set(node_ids) != set(col_ids):
        raise ValueError('Row and column identifiers do not match.')
    nodes = node_ids
    d = {}
    for rec in records:
        i = rec['values']['Unnamed: 0']
        d[i] = {}
        for j in col_ids:
            val_str = rec['values'][j]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid distance value for pair ({i},{j}): {val_str}')
            d[i][j] = val
    for i in nodes:
        for j in nodes:
            if j not in d[i]:
                raise ValueError(f'Missing distance for pair ({i},{j})')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in nodes for j in nodes], vtype=GRB.BINARY, name='')
    u = m.addVars([i for i in nodes if i != 'Depot'], vtype=GRB.INTEGER, lb=1, ub=len(nodes) - 1, name='')
    obj = gurobipy.quicksum((d[i][j] * x[i, j] for i in nodes for j in nodes))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in nodes)) == 1 for i in nodes), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in nodes)) == 1 for j in nodes), name='')
    n_minus_1 = len(nodes) - 1
    for i in nodes:
        if i == 'Depot':
            continue
        for j in nodes:
            if j == 'Depot' or i == j:
                continue
            m.addConstr(u[i] - u[j] + n_minus_1 * x[i, j] <= n_minus_1 - 1, name='mtz')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
