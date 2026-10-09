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
 'tables': [{'columns': ['Unnamed: 0',
                         'annual_training_hours',
                         'P1',
                         'manager_professional_association',
                         'manager_site_visit_count_2025_q4',
                         'P2',
                         'P3',
                         'operations_region'],
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
                                     'manager_professional_association': 'Construction',
                                     'manager_site_visit_count_2025_q4': '3',
                                     'operations_region': 'East'}},
                         {'source_row': 1,
                          'values': {'P1': '2800',
                                     'P2': '3300',
                                     'P3': '2900',
                                     'Unnamed: 0': 'MB',
                                     'annual_training_hours': '18',
                                     'manager_professional_association': 'Civil',
                                     'manager_site_visit_count_2025_q4': '18',
                                     'operations_region': 'East'}},
                         {'source_row': 2,
                          'values': {'P1': '2900',
                                     'P2': '3100',
                                     'P3': '3000',
                                     'Unnamed: 0': 'MC',
                                     'annual_training_hours': '12',
                                     'manager_professional_association': 'Civil',
                                     'manager_site_visit_count_2025_q4': '18',
                                     'operations_region': 'East'}}],
             'returned_rows': 3,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    df = CSVQA_FRAMES['file_0_view_0']
    managers = list(df['Unnamed: 0'].unique())
    project_cols = [col for col in ['P1', 'P2', 'P3'] if col in df.columns]
    projects = project_cols.copy()
    cost = {}
    for (_, row) in df.iterrows():
        manager = row['Unnamed: 0']
        cost[manager] = {}
        for project in projects:
            val = row[project]
            try:
                cij = float(val)
            except Exception:
                raise ValueError(f'Non-numeric or missing cost for manager {manager}, project {project}: {val}')
            cost[manager][project] = cij
    if len(managers) != len(projects):
        raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
    for manager in managers:
        for project in projects:
            if project not in cost[manager]:
                raise ValueError(f'Missing cost for manager {manager}, project {project}')
    m = gp.Model('manager_project_assignment')
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
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