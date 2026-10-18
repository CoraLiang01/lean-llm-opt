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
 'validation': {'fallback_reason': "Matrix ID column None is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column None is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Table with table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
if not records:
    raise ValueError("No records found in 'manager_project_costs.csv'.")
manager_col = 'Unnamed: 0'
project_cols = [col for col in table['columns'] if col != manager_col]
if not project_cols:
    raise ValueError("No project columns found in 'manager_project_costs.csv'.")
managers = [rec['values'][manager_col] for rec in records]
projects = project_cols
cost = {}
for rec in records:
    m = rec['values'][manager_col]
    for p in projects:
        try:
            c = float(rec['values'][p])
        except Exception:
            raise ValueError(f"Invalid or missing cost for manager '{m}', project '{p}'.")
        cost[m, p] = c
if len(managers) != len(projects):
    raise ValueError('Number of managers and projects must be equal for one-to-one assignment.')
m = gp.Model('AssignmentModel')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr, prj] * x_vars[mgr, prj] for mgr in managers for prj in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')