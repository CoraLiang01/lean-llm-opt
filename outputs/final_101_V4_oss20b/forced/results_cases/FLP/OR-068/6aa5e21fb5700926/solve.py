CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, a set of managers and a set of construction projects are given. Assigning '
          'manager i to project j incurs a cost that depends on the manager’s experience and expertise. These costs '
          'are provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each '
          'column corresponds to a project. The task is to determine a minimum-cost one-to-one assignment of managers '
          'to projects, such that each manager is assigned to exactly one project and each project is assigned to '
          'exactly one manager.',
 'relationships': [],
 'route': 'FLP',
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
    row_id_col = 'Unnamed: 0'
    if row_id_col not in columns:
        raise ValueError(f"Row identifier column '{row_id_col}' missing in table columns")
    project_cols = [c for c in columns if c != row_id_col]
    managers = []
    cost = {}
    for rec in table['records']:
        values = rec['values']
        mgr = values[row_id_col]
        if mgr in managers:
            raise ValueError(f"Duplicate manager identifier '{mgr}' in data")
        managers.append(mgr)
        cost[mgr] = {}
        for pj in project_cols:
            if pj not in values:
                raise ValueError(f"Missing cost for manager '{mgr}' and project '{pj}'")
            try:
                cost_val = float(values[pj])
            except ValueError:
                raise ValueError(f"Non-numeric cost value for manager '{mgr}', project '{pj}': {values[pj]}")
            cost[mgr][pj] = cost_val
    for mgr in managers:
        for pj in project_cols:
            if pj not in cost[mgr]:
                raise ValueError(f"Cost missing for manager '{mgr}', project '{pj}'")
    m = gurobipy.Model('assignment_problem')
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in managers for j in project_cols]
    x = m.addVars(keys, vtype=GRB.BINARY, name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in project_cols)) == 1 for i in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in managers)) == 1 for j in project_cols), name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in managers for j in project_cols)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()