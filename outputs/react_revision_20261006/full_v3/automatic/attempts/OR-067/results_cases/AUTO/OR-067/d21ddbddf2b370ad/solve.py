CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'AP',
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
 'validation': {'fallback_reason': "Matrix ID column 'column_header' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column 'column_header' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    records = table['records']
    manager_col = 'Unnamed: 0'
    project_cols = ['P1', 'P2', 'P3']
    M = [record['values'][manager_col] for record in records]
    P = project_cols
    c = {}
    for record in records:
        m = record['values'][manager_col]
        c[m] = {}
        for p in P:
            val = record['values'][p]
            try:
                c[m][p] = float(val)
            except Exception:
                raise ValueError(f'Invalid cost value for manager {m}, project {p}: {val}')
    for m in M:
        for p in P:
            if p not in c[m]:
                raise ValueError(f'Missing cost for manager {m}, project {p}')
    m = gp.Model()
    x_keys = [(mgr, proj) for mgr in M for proj in P]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[mgr][proj] * x[mgr, proj] for mgr in M for proj in P)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[mgr, proj] for proj in P)) == 1 for mgr in M), name='')
    m.addConstrs((gp.quicksum((x[mgr, proj] for mgr in M)) == 1 for proj in P), name='')
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