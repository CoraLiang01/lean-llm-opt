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
        raise RuntimeError('Required table file_0_view_0 not found in CSVQA_DATA.')
    records = table['records']
    manager_col = 'Unnamed: 0'
    project_cols = [col for col in table['columns'] if col != manager_col]
    M = [rec['values'][manager_col] for rec in records]
    P = project_cols
    c = {}
    for rec in records:
        m = rec['values'][manager_col]
        for p in P:
            val = rec['values'][p]
            if val is None or val == '':
                raise ValueError(f'Missing cost for manager {m}, project {p}')
            c[m, p] = float(val)
    for m in M:
        for p in P:
            if (m, p) not in c:
                raise ValueError(f'Missing cost coefficient for ({m}, {p})')
    m_model = gp.Model()
    x = m_model.addVars([(m, p) for m in M for p in P], vtype=GRB.BINARY, name='')
    m_model.setObjective(gp.quicksum((c[m, p] * x[m, p] for m in M for p in P)), GRB.MINIMIZE)
    for m in M:
        m_model.addConstr(gp.quicksum((x[m, p] for p in P)) == 1, name='mgr_assign')
    for p in P:
        m_model.addConstr(gp.quicksum((x[m, p] for m in M)) == 1, name='prj_assign')
    m_model.Params.MIPGap = 0.0001
    m_model.optimize()
    if m_model.Status == GRB.OPTIMAL:
        print(m_model.ObjVal)
        for v in m_model.getVars():
            print(v.VarName, v.X)
    else:
        print(m_model.Status)
    return m_model
m = solve_problem()