CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, there are multiple managers and construction projects, with each manager '
          'incurring different costs for different projects based on their expertise and experience. This cost data is '
          'recorded in the ‚Äúmanager_project_costs.csv‚Äù file, where each row represents a manager and each column '
          'indicates the cost for that manager to complete a specific project. The objective is to determine the most '
          'cost-effective assignment of managers to projects, ensuring that each project is assigned to a single '
          'manager and each manager is responsible for only one project. The goal is to minimize the total cost while '
          'satisfying these assignment constraints.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['manager_design_review_count_2025_q4',
                         'manager_client_meeting_count_2025_q4',
                         'manager_site_visit_count_2025_q3',
                         'manager_mentoring_session_count_2025_q4',
                         'operations_region',
                         'team_support_staff_count',
                         'manager_conference_attendance_count_2025_q4',
                         'manager_professional_seminar_count_2025_q4',
                         'Manager',
                         'Project 1 Cost',
                         'manager_report_delivery_channel',
                         'manager_training_format',
                         'Project 2 Cost',
                         'manager_site_visit_count_2025_q4',
                         'manager_procurement_inquiry_count_2025_q4',
                         'annual_inspection_count',
                         'Project 3 Cost',
                         'Project 4 Cost',
                         'manager_safety_briefing_count_2025_q4',
                         'manager_professional_association',
                         'Project 5 Cost',
                         'Project 6 Cost',
                         'annual_training_hours',
                         'Project 7 Cost'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Manager': 'Manager 1',
                                     'Project 1 Cost': '2972',
                                     'Project 2 Cost': '2727',
                                     'Project 3 Cost': '2795',
                                     'Project 4 Cost': '2922',
                                     'Project 5 Cost': '1302',
                                     'Project 6 Cost': '2489',
                                     'Project 7 Cost': '1533',
                                     'annual_inspection_count': '3',
                                     'annual_training_hours': '12',
                                     'manager_client_meeting_count_2025_q4': '14',
                                     'manager_conference_attendance_count_2025_q4': '3',
                                     'manager_design_review_count_2025_q4': '7',
                                     'manager_mentoring_session_count_2025_q4': '2',
                                     'manager_procurement_inquiry_count_2025_q4': '5',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '6',
                                     'manager_report_delivery_channel': 'Email',
                                     'manager_safety_briefing_count_2025_q4': '6',
                                     'manager_site_visit_count_2025_q3': '9',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'manager_training_format': 'Online',
                                     'operations_region': 'South',
                                     'team_support_staff_count': '6'}},
                         {'source_row': 1,
                          'values': {'Manager': 'Manager 2',
                                     'Project 1 Cost': '1094',
                                     'Project 2 Cost': '2158',
                                     'Project 3 Cost': '2990',
                                     'Project 4 Cost': '1844',
                                     'Project 5 Cost': '2887',
                                     'Project 6 Cost': '2021',
                                     'Project 7 Cost': '2288',
                                     'annual_inspection_count': '2',
                                     'annual_training_hours': '36',
                                     'manager_client_meeting_count_2025_q4': '7',
                                     'manager_conference_attendance_count_2025_q4': '0',
                                     'manager_design_review_count_2025_q4': '2',
                                     'manager_mentoring_session_count_2025_q4': '2',
                                     'manager_procurement_inquiry_count_2025_q4': '5',
                                     'manager_professional_association': 'Construction',
                                     'manager_professional_seminar_count_2025_q4': '6',
                                     'manager_report_delivery_channel': 'Meeting',
                                     'manager_safety_briefing_count_2025_q4': '2',
                                     'manager_site_visit_count_2025_q3': '3',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'manager_training_format': 'Classroom',
                                     'operations_region': 'East',
                                     'team_support_staff_count': '10'}},
                         {'source_row': 2,
                          'values': {'Manager': 'Manager 3',
                                     'Project 1 Cost': '2133',
                                     'Project 2 Cost': '1675',
                                     'Project 3 Cost': '2422',
                                     'Project 4 Cost': '2639',
                                     'Project 5 Cost': '1033',
                                     'Project 6 Cost': '2261',
                                     'Project 7 Cost': '1695',
                                     'annual_inspection_count': '6',
                                     'annual_training_hours': '48',
                                     'manager_client_meeting_count_2025_q4': '7',
                                     'manager_conference_attendance_count_2025_q4': '3',
                                     'manager_design_review_count_2025_q4': '5',
                                     'manager_mentoring_session_count_2025_q4': '6',
                                     'manager_procurement_inquiry_count_2025_q4': '5',
                                     'manager_professional_association': 'Construction',
                                     'manager_professional_seminar_count_2025_q4': '2',
                                     'manager_report_delivery_channel': 'Meeting',
                                     'manager_safety_briefing_count_2025_q4': '12',
                                     'manager_site_visit_count_2025_q3': '9',
                                     'manager_site_visit_count_2025_q4': '9',
                                     'manager_training_format': 'Classroom',
                                     'operations_region': 'North',
                                     'team_support_staff_count': '2'}},
                         {'source_row': 3,
                          'values': {'Manager': 'Manager 4',
                                     'Project 1 Cost': '1951',
                                     'Project 2 Cost': '2309',
                                     'Project 3 Cost': '2070',
                                     'Project 4 Cost': '2802',
                                     'Project 5 Cost': '2328',
                                     'Project 6 Cost': '1313',
                                     'Project 7 Cost': '2434',
                                     'annual_inspection_count': '1',
                                     'annual_training_hours': '48',
                                     'manager_client_meeting_count_2025_q4': '10',
                                     'manager_conference_attendance_count_2025_q4': '4',
                                     'manager_design_review_count_2025_q4': '5',
                                     'manager_mentoring_session_count_2025_q4': '6',
                                     'manager_procurement_inquiry_count_2025_q4': '16',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '1',
                                     'manager_report_delivery_channel': 'Portal',
                                     'manager_safety_briefing_count_2025_q4': '8',
                                     'manager_site_visit_count_2025_q3': '3',
                                     'manager_site_visit_count_2025_q4': '6',
                                     'manager_training_format': 'Classroom',
                                     'operations_region': 'South',
                                     'team_support_staff_count': '8'}},
                         {'source_row': 4,
                          'values': {'Manager': 'Manager 5',
                                     'Project 1 Cost': '1269',
                                     'Project 2 Cost': '2153',
                                     'Project 3 Cost': '1296',
                                     'Project 4 Cost': '2685',
                                     'Project 5 Cost': '2627',
                                     'Project 6 Cost': '1610',
                                     'Project 7 Cost': '1641',
                                     'annual_inspection_count': '3',
                                     'annual_training_hours': '12',
                                     'manager_client_meeting_count_2025_q4': '10',
                                     'manager_conference_attendance_count_2025_q4': '1',
                                     'manager_design_review_count_2025_q4': '10',
                                     'manager_mentoring_session_count_2025_q4': '6',
                                     'manager_procurement_inquiry_count_2025_q4': '12',
                                     'manager_professional_association': 'Civil',
                                     'manager_professional_seminar_count_2025_q4': '4',
                                     'manager_report_delivery_channel': 'Meeting',
                                     'manager_safety_briefing_count_2025_q4': '12',
                                     'manager_site_visit_count_2025_q3': '12',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'manager_training_format': 'Workshop',
                                     'operations_region': 'North',
                                     'team_support_staff_count': '4'}},
                         {'source_row': 5,
                          'values': {'Manager': 'Manager 6',
                                     'Project 1 Cost': '1220',
                                     'Project 2 Cost': '1192',
                                     'Project 3 Cost': '2907',
                                     'Project 4 Cost': '2622',
                                     'Project 5 Cost': '2595',
                                     'Project 6 Cost': '1261',
                                     'Project 7 Cost': '2384',
                                     'annual_inspection_count': '4',
                                     'annual_training_hours': '12',
                                     'manager_client_meeting_count_2025_q4': '7',
                                     'manager_conference_attendance_count_2025_q4': '0',
                                     'manager_design_review_count_2025_q4': '3',
                                     'manager_mentoring_session_count_2025_q4': '6',
                                     'manager_procurement_inquiry_count_2025_q4': '5',
                                     'manager_professional_association': 'General',
                                     'manager_professional_seminar_count_2025_q4': '2',
                                     'manager_report_delivery_channel': 'Email',
                                     'manager_safety_briefing_count_2025_q4': '8',
                                     'manager_site_visit_count_2025_q3': '18',
                                     'manager_site_visit_count_2025_q4': '18',
                                     'manager_training_format': 'Workshop',
                                     'operations_region': 'South',
                                     'team_support_staff_count': '8'}},
                         {'source_row': 6,
                          'values': {'Manager': 'Manager 7',
                                     'Project 1 Cost': '1286',
                                     'Project 2 Cost': '1659',
                                     'Project 3 Cost': '1179',
                                     'Project 4 Cost': '1348',
                                     'Project 5 Cost': '1420',
                                     'Project 6 Cost': '2862',
                                     'Project 7 Cost': '1959',
                                     'annual_inspection_count': '1',
                                     'annual_training_hours': '12',
                                     'manager_client_meeting_count_2025_q4': '7',
                                     'manager_conference_attendance_count_2025_q4': '1',
                                     'manager_design_review_count_2025_q4': '5',
                                     'manager_mentoring_session_count_2025_q4': '2',
                                     'manager_procurement_inquiry_count_2025_q4': '16',
                                     'manager_professional_association': 'Construction',
                                     'manager_professional_seminar_count_2025_q4': '3',
                                     'manager_report_delivery_channel': 'Portal',
                                     'manager_safety_briefing_count_2025_q4': '12',
                                     'manager_site_visit_count_2025_q3': '6',
                                     'manager_site_visit_count_2025_q4': '12',
                                     'manager_training_format': 'Workshop',
                                     'operations_region': 'East',
                                     'team_support_staff_count': '10'}}],
             'returned_rows': 7,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [7, 7], "
                                   "'expected_shape': [7, 7], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [7, 7], "
                                   "'expected_shape': [7, 7], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    import pandas as pd
    df = CSVQA_FRAMES['file_0_view_0']
    managers = list(df['Manager'])
    projects = ['Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost']
    if len(managers) != len(projects):
        raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal.')
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Manager']
        cost[manager] = {}
        for project in projects:
            val = row[project]
            try:
                cost[manager][project] = float(val)
            except Exception:
                raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': '{val}'")
    for manager in managers:
        for project in projects:
            if project not in cost[manager]:
                raise ValueError(f"Missing cost for manager '{manager}', project '{project}'")
    m = gp.Model('Original_RAG_AP')
    keys = [(i, j) for i in managers for j in projects]
    x_vars = m.addVars(keys, vtype=GRB.BINARY, name='')
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