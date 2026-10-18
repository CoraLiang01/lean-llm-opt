CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'TP',
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
table = CSVQA_DATA['tables'][0]
records = table['records']
columns = table['columns']
managers = [rec['values']['Unnamed: 0'] for rec in records]
projects = [col for col in columns if col != 'Unnamed: 0']
cost = {}
for rec in records:
    m = rec['values']['Unnamed: 0']
    cost[m] = {}
    for p in projects:
        cost[m][p] = float(rec['values'][p])
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {}
for i in managers:
    for j in projects:
        var = m.addVar(vtype=GRB.BINARY)
        x[i, j] = var
m.update()
obj = gp.quicksum((cost[i][j] * x[i, j] for i in managers for j in projects))
m.setObjective(obj, GRB.MINIMIZE)
for i in managers:
    m.addConstr(gp.quicksum((x[i, j] for j in projects)) == 1)
for j in projects:
    m.addConstr(gp.quicksum((x[i, j] for i in managers)) == 1)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print('Status:', m.status)