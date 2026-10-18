CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, there are multiple managers and construction projects, with each manager '
          'incurring different costs for different projects based on their expertise and experience. This cost data is '
          'recorded in the “manager_project_costs.csv” file, where each row represents a manager and each column '
          'indicates the cost for that manager to complete a specific project. The objective is to determine the most '
          'cost-effective assignment of managers to projects, ensuring that each project is assigned to a single '
          'manager and each manager is responsible for only one project. The goal is to minimize the total cost while '
          'satisfying these assignment constraints.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Manager',
                         'Project 1 Cost',
                         'Project 2 Cost',
                         'Project 3 Cost',
                         'Project 4 Cost',
                         'Project 5 Cost',
                         'Project 6 Cost',
                         'Project 7 Cost'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Manager': 'Manager 1',
                                     'Project 1 Cost': '2972',
                                     'Project 2 Cost': '2727',
                                     'Project 3 Cost': '2795',
                                     'Project 4 Cost': '2922',
                                     'Project 5 Cost': '1302',
                                     'Project 6 Cost': '2489',
                                     'Project 7 Cost': '1533'}},
                         {'source_row': 1,
                          'values': {'Manager': 'Manager 2',
                                     'Project 1 Cost': '1094',
                                     'Project 2 Cost': '2158',
                                     'Project 3 Cost': '2990',
                                     'Project 4 Cost': '1844',
                                     'Project 5 Cost': '2887',
                                     'Project 6 Cost': '2021',
                                     'Project 7 Cost': '2288'}},
                         {'source_row': 2,
                          'values': {'Manager': 'Manager 3',
                                     'Project 1 Cost': '2133',
                                     'Project 2 Cost': '1675',
                                     'Project 3 Cost': '2422',
                                     'Project 4 Cost': '2639',
                                     'Project 5 Cost': '1033',
                                     'Project 6 Cost': '2261',
                                     'Project 7 Cost': '1695'}},
                         {'source_row': 3,
                          'values': {'Manager': 'Manager 4',
                                     'Project 1 Cost': '1951',
                                     'Project 2 Cost': '2309',
                                     'Project 3 Cost': '2070',
                                     'Project 4 Cost': '2802',
                                     'Project 5 Cost': '2328',
                                     'Project 6 Cost': '1313',
                                     'Project 7 Cost': '2434'}},
                         {'source_row': 4,
                          'values': {'Manager': 'Manager 5',
                                     'Project 1 Cost': '1269',
                                     'Project 2 Cost': '2153',
                                     'Project 3 Cost': '1296',
                                     'Project 4 Cost': '2685',
                                     'Project 5 Cost': '2627',
                                     'Project 6 Cost': '1610',
                                     'Project 7 Cost': '1641'}},
                         {'source_row': 5,
                          'values': {'Manager': 'Manager 6',
                                     'Project 1 Cost': '1220',
                                     'Project 2 Cost': '1192',
                                     'Project 3 Cost': '2907',
                                     'Project 4 Cost': '2622',
                                     'Project 5 Cost': '2595',
                                     'Project 6 Cost': '1261',
                                     'Project 7 Cost': '2384'}},
                         {'source_row': 6,
                          'values': {'Manager': 'Manager 7',
                                     'Project 1 Cost': '1286',
                                     'Project 2 Cost': '1659',
                                     'Project 3 Cost': '1179',
                                     'Project 4 Cost': '1348',
                                     'Project 5 Cost': '1420',
                                     'Project 6 Cost': '2862',
                                     'Project 7 Cost': '1959'}}],
             'returned_rows': 7,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix ID column '' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column '' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Table with table_id 'file_0_view_0' not found.")
manager_col = 'Manager'
project_cost_cols = [col for col in table['columns'] if col.startswith('Project ') and col.endswith(' Cost')]
managers = []
projects = []
cost = {}
projects = [col.replace(' Cost', '') for col in project_cost_cols]
for rec in table['records']:
    m = rec['values'][manager_col]
    managers.append(m)
for m_idx, rec in enumerate(table['records']):
    m = rec['values'][manager_col]
    for p_idx, col in enumerate(project_cost_cols):
        p = col.replace(' Cost', '')
        try:
            c_val = int(rec['values'][col])
        except Exception:
            raise ValueError(f"Missing or invalid cost for manager '{m}', project '{p}'.")
        cost[m, p] = c_val
if len(managers) != len(projects):
    raise ValueError('Number of managers and projects must be equal for a one-to-one assignment.')
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f"Missing cost coefficient for manager '{m}', project '{p}'.")
m = gp.Model('manager_project_assignment')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr, prj] * x_vars[mgr, prj] for mgr in managers for prj in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')