CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, a set of managers and a set of construction projects are given. Assigning '
          'manager i to project j incurs a cost that depends on the manager’s experience and expertise. These costs '
          'are provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each '
          'column corresponds to a project. The task is to determine a minimum-cost one-to-one assignment of managers '
          'to projects, such that each manager is assigned to exactly one project and each project is assigned to '
          'exactly one manager.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'P1': '2216',
                                     'P2': '1911',
                                     'P3': '1661',
                                     'P4': '2122',
                                     'P5': '1442',
                                     'P6': '1442',
                                     'Unnamed: 0': 'MA'}},
                         {'source_row': 1,
                          'values': {'P1': '1100',
                                     'P2': '1271',
                                     'P3': '2764',
                                     'P4': '2557',
                                     'P5': '1036',
                                     'P6': '1036',
                                     'Unnamed: 0': 'MB'}},
                         {'source_row': 2,
                          'values': {'P1': '2827',
                                     'P2': '2784',
                                     'P3': '2206',
                                     'P4': '2216',
                                     'P5': '2677',
                                     'P6': '2677',
                                     'Unnamed: 0': 'MC'}},
                         {'source_row': 3,
                          'values': {'P1': '2627',
                                     'P2': '1273',
                                     'P3': '2610',
                                     'P4': '1957',
                                     'P5': '1594',
                                     'P6': '1594',
                                     'Unnamed: 0': 'MD'}},
                         {'source_row': 4,
                          'values': {'P1': '3359',
                                     'P2': '1003',
                                     'P3': '2554',
                                     'P4': '1706',
                                     'P5': '2065',
                                     'P6': '2065',
                                     'Unnamed: 0': 'ME'}},
                         {'source_row': 5,
                          'values': {'P1': '1579',
                                     'P2': '2289',
                                     'P3': '2368',
                                     'P4': '1922',
                                     'P5': '2740',
                                     'P6': '2740',
                                     'Unnamed: 0': 'MF'}}],
             'returned_rows': 6,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix ID column 'column header (P1-P6)' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column 'column header (P1-P6)' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Table 'file_0_view_0' not found in CSVQA_DATA.")
manager_ids = []
project_ids = []
cost = {}
for col in table['columns']:
    if col != 'Unnamed: 0':
        project_ids.append(col)
for rec in table['records']:
    row = rec['values']
    manager = row['Unnamed: 0']
    manager_ids.append(manager)
    for project in project_ids:
        try:
            cij = int(row[project])
        except Exception:
            raise ValueError(f'Missing or invalid cost for manager {manager}, project {project}')
        cost[manager, project] = cij
if len(manager_ids) != len(project_ids):
    raise ValueError('Number of managers and projects must be equal for one-to-one assignment.')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(manager_ids, project_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(gp.quicksum((cost[i, j] * x_vars[i, j] for i in manager_ids for j in project_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in project_ids)) == 1 for i in manager_ids), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in manager_ids)) == 1 for j in project_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')