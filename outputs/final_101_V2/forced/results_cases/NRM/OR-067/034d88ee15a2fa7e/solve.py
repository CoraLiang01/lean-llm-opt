CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}},
                         {'source_row': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}},
                         {'source_row': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}],
             'returned_rows': 3,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found in CSVQA_DATA.')
M = []
P = []
for col in table['columns']:
    if col != 'Unnamed: 0':
        P.append(col)
for rec in table['records']:
    m_id = rec['values']['Unnamed: 0']
    M.append(m_id)
M = list(M)
c = {}
for rec in table['records']:
    m_id = rec['values']['Unnamed: 0']
    for p_id in P:
        try:
            c_val = float(rec['values'][p_id])
        except Exception:
            raise ValueError(f'Missing or invalid cost for manager {m_id}, project {p_id}')
        c[m_id, p_id] = c_val
if set(M) != set((rec['values']['Unnamed: 0'] for rec in table['records'])):
    raise ValueError('Manager index set mismatch.')
if set(P) != set((col for col in table['columns'] if col != 'Unnamed: 0')):
    raise ValueError('Project index set mismatch.')
for m in M:
    for p in P:
        if (m, p) not in c:
            raise ValueError(f'Missing cost coefficient for ({m},{p})')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(M, P, vtype=GRB.BINARY, lb=0, name='')
m.setObjective(gp.quicksum((c[m_id, p_id] * x[m_id, p_id] for m_id in M for p_id in P)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[m_id, p_id] for p_id in P)) == 1 for m_id in M), name='')
m.addConstrs((gp.quicksum((x[m_id, p_id] for m_id in M)) == 1 for p_id in P), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')