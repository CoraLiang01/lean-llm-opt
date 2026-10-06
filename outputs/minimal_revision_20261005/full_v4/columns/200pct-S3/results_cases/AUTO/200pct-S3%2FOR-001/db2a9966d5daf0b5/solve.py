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
 'tables': [{'columns': ['previous_period_P3',
                         'Unnamed: 1',
                         'previous_period_P1',
                         'P1',
                         'two_periods_ago_assignment_status',
                         'previous_period_P2',
                         'three_periods_ago_assignment_status',
                         'P2',
                         'P3',
                         'four_periods_ago_assignment_status',
                         'previous_period_assignment_status',
                         'two_periods_ago_P1'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'P1': '3000',
                                     'P2': '3200',
                                     'P3': '3100',
                                     'Unnamed: 1': 'MA',
                                     'four_periods_ago_assignment_status': 'Reserved',
                                     'previous_period_P1': '2610',
                                     'previous_period_P2': '3340',
                                     'previous_period_P3': '3181',
                                     'previous_period_assignment_status': 'Completed',
                                     'three_periods_ago_assignment_status': 'Available',
                                     'two_periods_ago_P1': '3197',
                                     'two_periods_ago_assignment_status': 'Reserved'}},
                         {'source_row': 1,
                          'values': {'P1': '2800',
                                     'P2': '3300',
                                     'P3': '2900',
                                     'Unnamed: 1': 'MB',
                                     'four_periods_ago_assignment_status': 'Reserved',
                                     'previous_period_P1': '2358',
                                     'previous_period_P2': '2685',
                                     'previous_period_P3': '2493',
                                     'previous_period_assignment_status': 'Completed',
                                     'three_periods_ago_assignment_status': 'Available',
                                     'two_periods_ago_P1': '3162',
                                     'two_periods_ago_assignment_status': 'Completed'}},
                         {'source_row': 2,
                          'values': {'P1': '2900',
                                     'P2': '3100',
                                     'P3': '3000',
                                     'Unnamed: 1': 'MC',
                                     'four_periods_ago_assignment_status': 'Reserved',
                                     'previous_period_P1': '3130',
                                     'previous_period_P2': '3616',
                                     'previous_period_P3': '2732',
                                     'previous_period_assignment_status': 'Available',
                                     'three_periods_ago_assignment_status': 'Available',
                                     'two_periods_ago_P1': '2617',
                                     'two_periods_ago_assignment_status': 'Available'}}],
             'returned_rows': 3,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix ID column 'column header: P1, P2, P3' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column 'column header: P1, P2, P3' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise RuntimeError('Required table not found in CSVQA_DATA.')
    manager_col = 'Unnamed: 1'
    project_cols = ['P1', 'P2', 'P3']
    managers = []
    for rec in table['records']:
        managers.append(rec['values'][manager_col])
    projects = list(project_cols)
    cost = {}
    for rec in table['records']:
        m = rec['values'][manager_col]
        cost[m] = {}
        for p in projects:
            v = rec['values'][p]
            try:
                cost[m][p] = float(v)
            except Exception:
                raise ValueError(f'Invalid cost value for manager {m}, project {p}: {v}')
    for m in managers:
        for p in projects:
            if p not in cost[m]:
                raise ValueError(f'Missing cost for manager {m}, project {p}')
    m = gp.Model('manager_project_assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[man][proj] * x[man, proj] for man in managers for proj in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[man, proj] for proj in projects)) == 1 for man in managers), name='')
    m.addConstrs((gp.quicksum((x[man, proj] for man in managers)) == 1 for proj in projects), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()