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
 'tables': [{'columns': ['manager_professional_seminar_count_2025_q4',
                         'Unnamed: 1',
                         'annual_training_hours',
                         'P1',
                         'manager_professional_association',
                         'manager_site_visit_count_2025_q4',
                         'manager_report_delivery_channel',
                         'P2',
                         'P3',
                         'manager_training_format',
                         'operations_region',
                         'manager_client_meeting_count_2025_q4'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'P1': '3000',
                                     'P2': '3200',
                                     'P3': '3100',
                                     'Unnamed: 1': 'MA',
                                     'annual_training_hours': '18',
                                     'manager_client_meeting_count_2025_q4': '10',
                                     'manager_professional_association': 'Construction',
                                     'manager_professional_seminar_count_2025_q4': '3',
                                     'manager_report_delivery_channel': 'Portal',
                                     'manager_site_visit_count_2025_q4': '3',
                                     'manager_training_format': 'Classroom',
                                     'operations_region': 'East'}},
                         {'source_row': 1,
                          'values': {'P1': '2800',
                                     'P2': '3300',
                                     'P3': '2900',
                                     'Unnamed: 1': 'MB',
                                     'annual_training_hours': '18',
                                     'manager_client_meeting_count_2025_q4': '4',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '1',
                                     'manager_report_delivery_channel': 'Portal',
                                     'manager_site_visit_count_2025_q4': '18',
                                     'manager_training_format': 'Classroom',
                                     'operations_region': 'East'}},
                         {'source_row': 2,
                          'values': {'P1': '2900',
                                     'P2': '3100',
                                     'P3': '3000',
                                     'Unnamed: 1': 'MC',
                                     'annual_training_hours': '12',
                                     'manager_client_meeting_count_2025_q4': '20',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '1',
                                     'manager_report_delivery_channel': 'Meeting',
                                     'manager_site_visit_count_2025_q4': '18',
                                     'manager_training_format': 'Workshop',
                                     'operations_region': 'East'}}],
             'returned_rows': 3,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix ID column 'column header: P1, P2, P3' is not selected for 'file_0_view_0'",
                'planner_errors': ["Matrix ID column 'column header: P1, P2, P3' is not selected for 'file_0_view_0'"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = [{'manager': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}, {'manager': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}, {'manager': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}]
    managers = [row['manager'] for row in data]
    projects = ['P1', 'P2', 'P3']
    cost = {m: {p: int(row[p]) for p in projects} for (m, row) in zip(managers, data)}
    for m in managers:
        for p in projects:
            if p not in cost[m]:
                raise ValueError(f'Missing cost for manager {m}, project {p}')
    m = gp.Model('manager_project_assignment')
    x = m.addVars([(mgr, prj) for mgr in managers for prj in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[mgr][prj] * x[mgr, prj] for mgr in managers for prj in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
    m.addConstrs((gp.quicksum((x[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
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