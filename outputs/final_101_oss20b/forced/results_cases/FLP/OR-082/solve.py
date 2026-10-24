CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    name_to_index = {'Depot': 0, 'A': 1, 'B': 2, 'C': 3}
    index_to_name = {v: k for k, v in name_to_index.items()}
    nodes = list(name_to_index.values())
    d = {}
    for record in table['records']:
        row_name = record['values']['Unnamed: 0']
        if row_name not in name_to_index:
            continue
        i = name_to_index[row_name]
        for col_name in table['columns']:
            if col_name == 'Unnamed: 0':
                continue
            if col_name not in name_to_index:
                continue
            j = name_to_index[col_name]
            if i == j:
                continue
            val_str = record['values'][col_name]
            try:
                val = float(val_str) if '.' in val_str else int(val_str)
            except ValueError:
                raise ValueError(f'Invalid numeric value for d[{i}][{j}]: {val_str}')
            d[i, j] = val
    required_pairs = {(i, j) for i in nodes for j in nodes if i != j}
    missing_pairs = required_pairs - d.keys()
    if missing_pairs:
        raise ValueError(f'Missing distance data for pairs: {missing_pairs}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars([i for i in nodes if i != 0], vtype=GRB.INTEGER, lb=1, ub=3, name='')
    for i in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
    for j in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'in_{j}')
    for i in nodes:
        if i == 0:
            continue
        for j in nodes:
            if j == 0 or i == j:
                continue
            m.addConstr(u[i] - u[j] + 4 * x[i, j] <= 3, name=f'mtz_{i}_{j}')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
