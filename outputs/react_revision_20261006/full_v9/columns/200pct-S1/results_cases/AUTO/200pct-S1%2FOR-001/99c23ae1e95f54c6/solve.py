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
 'tables': [{'columns': ['record_view_count',
                         'Unnamed: 1',
                         'archive_revision_number',
                         'P1',
                         'archive_storage_medium',
                         'document_page_count',
                         'document_template_family',
                         'P2',
                         'P3',
                         'record_label_font',
                         'record_display_theme',
                         'archive_batch_number'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'P1': '3000',
                                     'P2': '3200',
                                     'P3': '3100',
                                     'Unnamed: 1': 'MA',
                                     'archive_batch_number': '305',
                                     'archive_revision_number': '2',
                                     'archive_storage_medium': 'Digital',
                                     'document_page_count': '6',
                                     'document_template_family': 'Compact',
                                     'record_display_theme': 'Amber',
                                     'record_label_font': 'Helvetica',
                                     'record_view_count': '27'}},
                         {'source_row': 1,
                          'values': {'P1': '2800',
                                     'P2': '3300',
                                     'P3': '2900',
                                     'Unnamed: 1': 'MB',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '2',
                                     'archive_storage_medium': 'Hybrid',
                                     'document_page_count': '8',
                                     'document_template_family': 'Landscape',
                                     'record_display_theme': 'Amber',
                                     'record_label_font': 'Calibri',
                                     'record_view_count': '76'}},
                         {'source_row': 2,
                          'values': {'P1': '2900',
                                     'P2': '3100',
                                     'P3': '3000',
                                     'Unnamed: 1': 'MC',
                                     'archive_batch_number': '303',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Digital',
                                     'document_page_count': '4',
                                     'document_template_family': 'Landscape',
                                     'record_display_theme': 'Azure',
                                     'record_label_font': 'Calibri',
                                     'record_view_count': '58'}}],
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
    frame = CSVQA_FRAMES['file_0_view_0']
    managers = []
    projects = ['P1', 'P2', 'P3']
    cost = {}
    for (source_row, row) in frame.iterrows():
        manager = row['Unnamed: 1']
        managers.append(manager)
        cost[manager] = {}
        for project in projects:
            try:
                cost[manager][project] = float(row[project])
            except Exception:
                raise ValueError(f'Missing or invalid cost for manager {manager}, project {project}')
    if len(managers) != len(projects):
        raise ValueError('Number of managers and projects must be equal for one-to-one assignment.')
    for manager in managers:
        for project in projects:
            if project not in cost[manager]:
                raise ValueError(f'Missing cost for manager {manager}, project {project}')
    m = gp.Model('manager_project_assignment')
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[manager][project] * x_vars[manager, project] for manager in managers for project in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[manager, project] for project in projects)) == 1 for manager in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[manager, project] for manager in managers)) == 1 for project in projects), name='')
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