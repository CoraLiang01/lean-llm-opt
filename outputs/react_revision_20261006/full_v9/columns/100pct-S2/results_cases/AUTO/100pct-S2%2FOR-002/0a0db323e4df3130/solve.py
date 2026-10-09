CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, a set of managers and a set of construction projects are given. Assigning '
          'manager i to project j incurs a cost that depends on the manager‚Äôs experience and expertise. These costs '
          'are provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each '
          'column corresponds to a project. The task is to determine a minimum-cost one-to-one assignment of managers '
          'to projects, such that each manager is assigned to exactly one project and each project is assigned to '
          'exactly one manager.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Unnamed: 0',
                         'team_support_staff_count',
                         'P1',
                         'P2',
                         'annual_training_hours',
                         'P3',
                         'P4',
                         'manager_site_visit_count_2025_q4',
                         'annual_inspection_count',
                         'P5',
                         'manager_professional_seminar_count_2025_q4',
                         'operations_region',
                         'manager_professional_association',
                         'P6'],
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
                                     'Unnamed: 0': 'MA',
                                     'annual_inspection_count': '4',
                                     'annual_training_hours': '24',
                                     'manager_professional_association': 'General',
                                     'manager_professional_seminar_count_2025_q4': '1',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'operations_region': 'West',
                                     'team_support_staff_count': '6'}},
                         {'source_row': 1,
                          'values': {'P1': '1100',
                                     'P2': '1271',
                                     'P3': '2764',
                                     'P4': '2557',
                                     'P5': '1036',
                                     'P6': '1036',
                                     'Unnamed: 0': 'MB',
                                     'annual_inspection_count': '4',
                                     'annual_training_hours': '18',
                                     'manager_professional_association': 'General',
                                     'manager_professional_seminar_count_2025_q4': '4',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'operations_region': 'East',
                                     'team_support_staff_count': '6'}},
                         {'source_row': 2,
                          'values': {'P1': '2827',
                                     'P2': '2784',
                                     'P3': '2206',
                                     'P4': '2216',
                                     'P5': '2677',
                                     'P6': '2677',
                                     'Unnamed: 0': 'MC',
                                     'annual_inspection_count': '6',
                                     'annual_training_hours': '36',
                                     'manager_professional_association': 'General',
                                     'manager_professional_seminar_count_2025_q4': '2',
                                     'manager_site_visit_count_2025_q4': '9',
                                     'operations_region': 'East',
                                     'team_support_staff_count': '4'}},
                         {'source_row': 3,
                          'values': {'P1': '2627',
                                     'P2': '1273',
                                     'P3': '2610',
                                     'P4': '1957',
                                     'P5': '1594',
                                     'P6': '1594',
                                     'Unnamed: 0': 'MD',
                                     'annual_inspection_count': '2',
                                     'annual_training_hours': '48',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '2',
                                     'manager_site_visit_count_2025_q4': '6',
                                     'operations_region': 'West',
                                     'team_support_staff_count': '2'}},
                         {'source_row': 4,
                          'values': {'P1': '3359',
                                     'P2': '1003',
                                     'P3': '2554',
                                     'P4': '1706',
                                     'P5': '2065',
                                     'P6': '2065',
                                     'Unnamed: 0': 'ME',
                                     'annual_inspection_count': '1',
                                     'annual_training_hours': '18',
                                     'manager_professional_association': 'General',
                                     'manager_professional_seminar_count_2025_q4': '2',
                                     'manager_site_visit_count_2025_q4': '6',
                                     'operations_region': 'East',
                                     'team_support_staff_count': '6'}},
                         {'source_row': 5,
                          'values': {'P1': '1579',
                                     'P2': '2289',
                                     'P3': '2368',
                                     'P4': '1922',
                                     'P5': '2740',
                                     'P6': '2740',
                                     'Unnamed: 0': 'MF',
                                     'annual_inspection_count': '1',
                                     'annual_training_hours': '48',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '2',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'operations_region': 'East',
                                     'team_support_staff_count': '10'}}],
             'returned_rows': 6,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix ID column ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'] is not selected for "
                                   "'file_0_view_0'",
                'planner_errors': ["Matrix ID column ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'] is not selected for "
                                   "'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    managers = list(frame['Unnamed: 0'])
    projects = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    cost = {}
    for (_, row) in frame.iterrows():
        i = row['Unnamed: 0']
        cost[i] = {}
        for j in projects:
            cost[i][j] = float(row[j])
    if len(managers) != len(projects):
        raise ValueError('Number of managers and projects must be equal for one-to-one assignment.')
    for i in managers:
        for j in projects:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for manager {i}, project {j}')
    m = gp.Model()
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
m = solve_problem(CSVQA_FRAMES)