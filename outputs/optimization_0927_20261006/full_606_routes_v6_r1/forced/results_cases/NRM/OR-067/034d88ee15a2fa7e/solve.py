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
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_order_matches': True, 'column_order_matches': False}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_order_matches': True, 'column_order_matches': False}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = [r['values'] for r in CSVQA_DATA['tables'][0]['records']]
manager_col = 'Unnamed: 0'
project_cols = ['P1', 'P2', 'P3']
M = [row[manager_col] for row in table]
P = project_cols
c = {}
for row in table:
    m = row[manager_col]
    for p in P:
        try:
            c[m, p] = float(row[p])
        except Exception:
            raise ValueError(f'Missing or invalid cost for manager {m}, project {p}')
if len(M) != len(P):
    raise ValueError('Number of managers and projects must be equal for one-to-one assignment.')
for m in M:
    for p in P:
        if (m, p) not in c:
            raise ValueError(f'Missing cost for manager {m}, project {p}')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(M, P, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[m_key, p_key] * x_vars[m_key, p_key] for m_key in M for p_key in P)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[m_key, p_key] for p_key in P)) == 1 for m_key in M), name='')
m.addConstrs((gp.quicksum((x_vars[m_key, p_key] for m_key in M)) == 1 for p_key in P), name='')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')