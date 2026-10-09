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
 'tables': [{'columns': ['previous_period_assignment_status',
                         'previous_period_Project_2_Cost',
                         'previous_period_Project_5_Cost',
                         'Manager',
                         'Project 1 Cost',
                         'Project 2 Cost',
                         'previous_period_Project_4_Cost',
                         'previous_period_Project_3_Cost',
                         'Project 3 Cost',
                         'Project 4 Cost',
                         'previous_period_Project_6_Cost',
                         'two_periods_ago_assignment_status',
                         'Project 5 Cost',
                         'Project 6 Cost',
                         'previous_period_Project_1_Cost',
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
                                     'previous_period_Project_1_Cost': '2702',
                                     'previous_period_Project_2_Cost': '2308',
                                     'previous_period_Project_3_Cost': '3045',
                                     'previous_period_Project_4_Cost': '2788',
                                     'previous_period_Project_5_Cost': '1090',
                                     'previous_period_Project_6_Cost': '2236',
                                     'previous_period_assignment_status': 'Completed',
                                     'two_periods_ago_assignment_status': 'Available'}},
                         {'source_row': 1,
                          'values': {'Manager': 'Manager 2',
                                     'Project 1 Cost': '1094',
                                     'Project 2 Cost': '2158',
                                     'Project 3 Cost': '2990',
                                     'Project 4 Cost': '1844',
                                     'Project 5 Cost': '2887',
                                     'Project 6 Cost': '2021',
                                     'Project 7 Cost': '2288',
                                     'previous_period_Project_1_Cost': '1113',
                                     'previous_period_Project_2_Cost': '2203',
                                     'previous_period_Project_3_Cost': '2645',
                                     'previous_period_Project_4_Cost': '2140',
                                     'previous_period_Project_5_Cost': '3366',
                                     'previous_period_Project_6_Cost': '1916',
                                     'previous_period_assignment_status': 'Reserved',
                                     'two_periods_ago_assignment_status': 'Available'}},
                         {'source_row': 2,
                          'values': {'Manager': 'Manager 3',
                                     'Project 1 Cost': '2133',
                                     'Project 2 Cost': '1675',
                                     'Project 3 Cost': '2422',
                                     'Project 4 Cost': '2639',
                                     'Project 5 Cost': '1033',
                                     'Project 6 Cost': '2261',
                                     'Project 7 Cost': '1695',
                                     'previous_period_Project_1_Cost': '2482',
                                     'previous_period_Project_2_Cost': '1853',
                                     'previous_period_Project_3_Cost': '2088',
                                     'previous_period_Project_4_Cost': '2491',
                                     'previous_period_Project_5_Cost': '889',
                                     'previous_period_Project_6_Cost': '1949',
                                     'previous_period_assignment_status': 'Available',
                                     'two_periods_ago_assignment_status': 'Available'}},
                         {'source_row': 3,
                          'values': {'Manager': 'Manager 4',
                                     'Project 1 Cost': '1951',
                                     'Project 2 Cost': '2309',
                                     'Project 3 Cost': '2070',
                                     'Project 4 Cost': '2802',
                                     'Project 5 Cost': '2328',
                                     'Project 6 Cost': '1313',
                                     'Project 7 Cost': '2434',
                                     'previous_period_Project_1_Cost': '2104',
                                     'previous_period_Project_2_Cost': '1942',
                                     'previous_period_Project_3_Cost': '1790',
                                     'previous_period_Project_4_Cost': '3111',
                                     'previous_period_Project_5_Cost': '2487',
                                     'previous_period_Project_6_Cost': '1496',
                                     'previous_period_assignment_status': 'Completed',
                                     'two_periods_ago_assignment_status': 'Assigned'}},
                         {'source_row': 4,
                          'values': {'Manager': 'Manager 5',
                                     'Project 1 Cost': '1269',
                                     'Project 2 Cost': '2153',
                                     'Project 3 Cost': '1296',
                                     'Project 4 Cost': '2685',
                                     'Project 5 Cost': '2627',
                                     'Project 6 Cost': '1610',
                                     'Project 7 Cost': '1641',
                                     'previous_period_Project_1_Cost': '1300',
                                     'previous_period_Project_2_Cost': '2550',
                                     'previous_period_Project_3_Cost': '1177',
                                     'previous_period_Project_4_Cost': '3016',
                                     'previous_period_Project_5_Cost': '2545',
                                     'previous_period_Project_6_Cost': '1742',
                                     'previous_period_assignment_status': 'Reserved',
                                     'two_periods_ago_assignment_status': 'Assigned'}},
                         {'source_row': 5,
                          'values': {'Manager': 'Manager 6',
                                     'Project 1 Cost': '1220',
                                     'Project 2 Cost': '1192',
                                     'Project 3 Cost': '2907',
                                     'Project 4 Cost': '2622',
                                     'Project 5 Cost': '2595',
                                     'Project 6 Cost': '1261',
                                     'Project 7 Cost': '2384',
                                     'previous_period_Project_1_Cost': '1244',
                                     'previous_period_Project_2_Cost': '1171',
                                     'previous_period_Project_3_Cost': '2844',
                                     'previous_period_Project_4_Cost': '2985',
                                     'previous_period_Project_5_Cost': '2998',
                                     'previous_period_Project_6_Cost': '1012',
                                     'previous_period_assignment_status': 'Available',
                                     'two_periods_ago_assignment_status': 'Available'}},
                         {'source_row': 6,
                          'values': {'Manager': 'Manager 7',
                                     'Project 1 Cost': '1286',
                                     'Project 2 Cost': '1659',
                                     'Project 3 Cost': '1179',
                                     'Project 4 Cost': '1348',
                                     'Project 5 Cost': '1420',
                                     'Project 6 Cost': '2862',
                                     'Project 7 Cost': '1959',
                                     'previous_period_Project_1_Cost': '1245',
                                     'previous_period_Project_2_Cost': '1967',
                                     'previous_period_Project_3_Cost': '1176',
                                     'previous_period_Project_4_Cost': '1263',
                                     'previous_period_Project_5_Cost': '1656',
                                     'previous_period_Project_6_Cost': '2638',
                                     'previous_period_assignment_status': 'Available',
                                     'two_periods_ago_assignment_status': 'Reserved'}}],
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

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7']
    projects = ['Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost']
    cost = {}
    for (_, row) in frame.iterrows():
        manager = row['Manager']
        cost[manager] = {}
        for project in projects:
            try:
                cost[manager][project] = float(row[project])
            except Exception:
                raise ValueError(f"Missing or invalid cost for manager '{manager}', project '{project}'")
    if set(cost.keys()) != set(managers):
        raise ValueError('Mismatch in manager identifiers between data and index set')
    for manager in managers:
        if set(cost[manager].keys()) != set(projects):
            raise ValueError(f"Mismatch in project identifiers for manager '{manager}'")
    m = gp.Model('assignment_problem')
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