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
 'tables': [{'columns': ['Unnamed: 0', 'annual_training_hours', 'P1', 'P2', 'P3', 'operations_region'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'P1': '3000',
                                     'P2': '3200',
                                     'P3': '3100',
                                     'Unnamed: 0': 'MA',
                                     'annual_training_hours': '18',
                                     'operations_region': 'East'}},
                         {'source_row': 1,
                          'values': {'P1': '2800',
                                     'P2': '3300',
                                     'P3': '2900',
                                     'Unnamed: 0': 'MB',
                                     'annual_training_hours': '18',
                                     'operations_region': 'East'}},
                         {'source_row': 2,
                          'values': {'P1': '2900',
                                     'P2': '3100',
                                     'P3': '3000',
                                     'Unnamed: 0': 'MC',
                                     'annual_training_hours': '12',
                                     'operations_region': 'East'}}],
             'returned_rows': 3,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix ID column 'column header: P1, P2, P3' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column 'column header: P1, P2, P3' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    frame = CSVQA_FRAMES['file_0_view_0']
    managers = list(frame['Unnamed: 0'])
    projects = ['P1', 'P2', 'P3']
    cost = {}
    for (idx, row) in frame.iterrows():
        m = row['Unnamed: 0']
        cost[m] = {}
        for p in projects:
            try:
                cost[m][p] = float(row[p])
            except Exception:
                raise ValueError(f'Missing or invalid cost for manager {m}, project {p}')
    if set(cost.keys()) != set(managers):
        raise ValueError('Mismatch in manager keys')
    for m in managers:
        if set(cost[m].keys()) != set(projects):
            raise ValueError(f'Mismatch in project keys for manager {m}')
    m = gp.Model('manager_project_assignment')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[mgr][prj] * x_vars[mgr, prj] for mgr in managers for prj in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
    m.optimize()
    return m
m = solve_problem()