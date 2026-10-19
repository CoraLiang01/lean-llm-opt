CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'manager_project_costs.csv'],
             'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}},
                         {'source_row': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}},
                         {'source_row': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', []):
        table = t
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found')
columns = table['columns']
projects = columns[1:]
records = table['records']
costs = {}
managers = []
for rec in records:
    mgr = rec['values']['Unnamed: 0']
    managers.append(mgr)
    costs[mgr] = {}
    for proj in projects:
        costs[mgr][proj] = int(rec['values'][proj])
m = gp.Model('assignment')
x = {}
for mgr in managers:
    x[mgr] = {}
    for proj in projects:
        var = m.addVar(vtype=GRB.BINARY)
        x[mgr][proj] = var
m.setObjective(gp.quicksum((costs[mgr][proj] * x[mgr][proj] for mgr in managers for proj in projects)), GRB.MINIMIZE)
for mgr in managers:
    m.addConstr(gp.quicksum((x[mgr][proj] for proj in projects)) == 1)
for proj in projects:
    m.addConstr(gp.quicksum((x[mgr][proj] for mgr in managers)) == 1)
m.setParam('MIPGap', 1e-06)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} = {v.X}')
else:
    print(f'Status: {m.Status}')