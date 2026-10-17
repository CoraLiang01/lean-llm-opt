CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, a set of managers and a set of construction projects are given. Assigning '
          'manager i to project j incurs a cost that depends on the manager’s experience and expertise. These costs '
          'are provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each '
          'column corresponds to a project. The task is to determine a minimum-cost one-to-one assignment of managers '
          'to projects, such that each manager is assigned to exactly one project and each project is assigned to '
          'exactly one manager.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    columns = table.get('columns', [])
    records = table.get('records', [])
    managers = [rec['values']['Unnamed: 0'] for rec in records]
    projects = [col for col in columns if col != 'Unnamed: 0']
    if len(set(managers)) != len(managers):
        raise ValueError('Duplicate manager identifiers detected')
    if len(set(projects)) != len(projects):
        raise ValueError('Duplicate project identifiers detected')
    costs = {}
    for rec in records:
        m = rec['values']['Unnamed: 0']
        for p in projects:
            if p not in rec['values']:
                raise ValueError(f'Missing cost for manager {m} and project {p}')
            val = rec['values'][p]
            try:
                cost = int(val)
            except Exception as e:
                raise ValueError(f"Invalid cost value '{val}' for manager {m}, project {p}") from e
            costs[m, p] = cost
    for m in managers:
        for p in projects:
            if (m, p) not in costs:
                raise ValueError(f'Missing cost entry for manager {m} and project {p}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(m, p) for m in managers for p in projects], vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((costs[m, p] * x[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[m, p] for p in projects)) == 1 for m in managers), name='')
    model.addConstrs((gurobipy.quicksum((x[m, p] for m in managers)) == 1 for p in projects), name='')
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')