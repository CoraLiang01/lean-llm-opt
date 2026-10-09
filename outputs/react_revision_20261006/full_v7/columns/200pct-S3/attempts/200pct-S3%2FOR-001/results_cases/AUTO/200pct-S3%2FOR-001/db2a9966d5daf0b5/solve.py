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
 'validation': {'fallback_reason': "Matrix ID column 'column_header' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column 'column_header' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    df = CSVQA_FRAMES['file_0_view_0']
    managers = list(df['Unnamed: 1'])
    projects = ['P1', 'P2', 'P3']
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Unnamed: 1']
        cost[manager] = {}
        for project in projects:
            try:
                cij = float(row[project])
            except Exception:
                raise ValueError(f'Missing or invalid cost for manager {manager}, project {project}')
            cost[manager][project] = cij
    if set(cost.keys()) != set(managers):
        raise ValueError('Mismatch in manager keys')
    for manager in managers:
        if set(cost[manager].keys()) != set(projects):
            raise ValueError(f'Mismatch in project keys for manager {manager}')
    m = gp.Model()
    x_vars = m.addVars([(i, j) for i in managers for j in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()