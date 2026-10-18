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
 'tables': [{'aliases': ['file_0_view_0', 'manager_project_costs.csv'],
             'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'],
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
import gurobipy as gp
from gurobipy import GRB
records = CSVQA_DATA['tables'][0]['records']
managers = [rec['values']['Unnamed: 0'] for rec in records]
projects = [col for col in CSVQA_DATA['tables'][0]['columns'] if col != 'Unnamed: 0']
cost = {}
for rec in records:
    m_id = rec['values']['Unnamed: 0']
    cost[m_id] = {}
    for p in projects:
        cost[m_id][p] = float(rec['values'][p])
m = gp.Model('assignment')
m.setParam('MIPGap', 1e-06)
x = {(m_id, p): m.addVar(vtype=GRB.BINARY) for m_id in managers for p in projects}
for m_id in managers:
    m.addConstr(gp.quicksum((x[m_id, p] for p in projects)) == 1)
for p in projects:
    m.addConstr(gp.quicksum((x[m_id, p] for m_id in managers)) == 1)
m.setObjective(gp.quicksum((cost[m_id][p] * x[m_id, p] for m_id in managers for p in projects)), GRB.MINIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.objVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.status)