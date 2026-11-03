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
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}},
                         {'source_row': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}},
                         {'source_row': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    columns = table['columns']
    if 'Unnamed: 0' not in columns:
        raise ValueError("Column 'Unnamed: 0' not found in table columns")
    project_columns = [c for c in columns if c != 'Unnamed: 0']
    managers = set()
    for rec in table['records']:
        mgr = rec['values']['Unnamed: 0']
        managers.add(mgr)
    managers = sorted(managers)
    projects = sorted(project_columns)
    cost = {}
    for rec in table['records']:
        mgr = rec['values']['Unnamed: 0']
        for p in projects:
            val_str = rec['values'].get(p)
            if val_str is None:
                raise ValueError(f'Missing cost for manager {mgr} and project {p}')
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f"Non-numeric cost '{val_str}' for manager {mgr} and project {p}")
            cost[mgr, p] = val
    for m in managers:
        for p in projects:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost entry for manager {m} and project {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    var_keys = [(mgr, p) for mgr in managers for p in projects]
    x = m.addVars(var_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[mgr, p] * x[mgr, p] for mgr in managers for p in projects)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[mgr, p] for p in projects)) == 1 for mgr in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[mgr, p] for mgr in managers)) == 1 for p in projects), name='')
    m.optimize()
    return m
m = solve_problem()