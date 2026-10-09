CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A set of managers and a set of construction projects are given. Assigning manager i to project j incurs a '
          "cost c_ij, reflecting the manager's suitability, experience, and expertise for that project. The cost "
          'matrix is provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and '
          'each column corresponds to a project.\n'
          '\n'
          'The task is to determine a minimum-cost one-to-one assignment of managers to projects. Each manager must be '
          'assigned to exactly one project, and each project must be assigned to exactly one manager. The objective is '
          'to minimize the total assignment cost over all manager-project pairs.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Manager',
                         'Project 1 Cost',
                         'Project 2 Cost',
                         'Project 3 Cost',
                         'Project 4 Cost',
                         'Project 5 Cost',
                         'Project 6 Cost',
                         'Project 7 Cost',
                         'Project 8 Cost',
                         'Project 9 Cost',
                         'Project 10 Cost',
                         'Project 11 Cost'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'Manager': 'Manager 1',
                                     'Project 1 Cost': '708',
                                     'Project 10 Cost': '3167',
                                     'Project 11 Cost': '1711',
                                     'Project 2 Cost': '1948',
                                     'Project 3 Cost': '2424',
                                     'Project 4 Cost': '1068',
                                     'Project 5 Cost': '729',
                                     'Project 6 Cost': '199',
                                     'Project 7 Cost': '1651',
                                     'Project 8 Cost': '3174',
                                     'Project 9 Cost': '3211'}},
                         {'source_row': 1,
                          'values': {'Manager': 'Manager 2',
                                     'Project 1 Cost': '1700',
                                     'Project 10 Cost': '2603',
                                     'Project 11 Cost': '1822',
                                     'Project 2 Cost': '2670',
                                     'Project 3 Cost': '1883',
                                     'Project 4 Cost': '2534',
                                     'Project 5 Cost': '1429',
                                     'Project 6 Cost': '1173',
                                     'Project 7 Cost': '777',
                                     'Project 8 Cost': '248',
                                     'Project 9 Cost': '1704'}},
                         {'source_row': 2,
                          'values': {'Manager': 'Manager 3',
                                     'Project 1 Cost': '160',
                                     'Project 10 Cost': '2502',
                                     'Project 11 Cost': '2595',
                                     'Project 2 Cost': '755',
                                     'Project 3 Cost': '3477',
                                     'Project 4 Cost': '3122',
                                     'Project 5 Cost': '2968',
                                     'Project 6 Cost': '3023',
                                     'Project 7 Cost': '1417',
                                     'Project 8 Cost': '254',
                                     'Project 9 Cost': '3175'}},
                         {'source_row': 3,
                          'values': {'Manager': 'Manager 4',
                                     'Project 1 Cost': '2213',
                                     'Project 10 Cost': '1954',
                                     'Project 11 Cost': '1805',
                                     'Project 2 Cost': '1008',
                                     'Project 3 Cost': '411',
                                     'Project 4 Cost': '1199',
                                     'Project 5 Cost': '418',
                                     'Project 6 Cost': '1000',
                                     'Project 7 Cost': '3148',
                                     'Project 8 Cost': '1724',
                                     'Project 9 Cost': '1984'}},
                         {'source_row': 4,
                          'values': {'Manager': 'Manager 5',
                                     'Project 1 Cost': '198',
                                     'Project 10 Cost': '270',
                                     'Project 11 Cost': '1893',
                                     'Project 2 Cost': '1721',
                                     'Project 3 Cost': '1318',
                                     'Project 4 Cost': '3194',
                                     'Project 5 Cost': '3036',
                                     'Project 6 Cost': '2938',
                                     'Project 7 Cost': '3298',
                                     'Project 8 Cost': '3332',
                                     'Project 9 Cost': '1806'}},
                         {'source_row': 5,
                          'values': {'Manager': 'Manager 6',
                                     'Project 1 Cost': '2375',
                                     'Project 10 Cost': '2696',
                                     'Project 11 Cost': '2217',
                                     'Project 2 Cost': '1804',
                                     'Project 3 Cost': '3174',
                                     'Project 4 Cost': '1607',
                                     'Project 5 Cost': '2168',
                                     'Project 6 Cost': '1642',
                                     'Project 7 Cost': '970',
                                     'Project 8 Cost': '3433',
                                     'Project 9 Cost': '1528'}},
                         {'source_row': 6,
                          'values': {'Manager': 'Manager 7',
                                     'Project 1 Cost': '2400',
                                     'Project 10 Cost': '2762',
                                     'Project 11 Cost': '577',
                                     'Project 2 Cost': '211',
                                     'Project 3 Cost': '1172',
                                     'Project 4 Cost': '425',
                                     'Project 5 Cost': '1222',
                                     'Project 6 Cost': '287',
                                     'Project 7 Cost': '653',
                                     'Project 8 Cost': '1466',
                                     'Project 9 Cost': '479'}},
                         {'source_row': 7,
                          'values': {'Manager': 'Manager 8',
                                     'Project 1 Cost': '272',
                                     'Project 10 Cost': '3260',
                                     'Project 11 Cost': '2981',
                                     'Project 2 Cost': '2574',
                                     'Project 3 Cost': '413',
                                     'Project 4 Cost': '202',
                                     'Project 5 Cost': '1220',
                                     'Project 6 Cost': '2392',
                                     'Project 7 Cost': '410',
                                     'Project 8 Cost': '2250',
                                     'Project 9 Cost': '2272'}},
                         {'source_row': 8,
                          'values': {'Manager': 'Manager 9',
                                     'Project 1 Cost': '2844',
                                     'Project 10 Cost': '114',
                                     'Project 11 Cost': '2161',
                                     'Project 2 Cost': '2775',
                                     'Project 3 Cost': '357',
                                     'Project 4 Cost': '2601',
                                     'Project 5 Cost': '1627',
                                     'Project 6 Cost': '125',
                                     'Project 7 Cost': '1029',
                                     'Project 8 Cost': '1354',
                                     'Project 9 Cost': '2280'}},
                         {'source_row': 9,
                          'values': {'Manager': 'Manager 10',
                                     'Project 1 Cost': '1222',
                                     'Project 10 Cost': '1873',
                                     'Project 11 Cost': '1185',
                                     'Project 2 Cost': '296',
                                     'Project 3 Cost': '3375',
                                     'Project 4 Cost': '352',
                                     'Project 5 Cost': '2167',
                                     'Project 6 Cost': '2202',
                                     'Project 7 Cost': '3139',
                                     'Project 8 Cost': '2526',
                                     'Project 9 Cost': '767'}},
                         {'source_row': 10,
                          'values': {'Manager': 'Manager 11',
                                     'Project 1 Cost': '2661',
                                     'Project 10 Cost': '3003',
                                     'Project 11 Cost': '2161',
                                     'Project 2 Cost': '887',
                                     'Project 3 Cost': '455',
                                     'Project 4 Cost': '2552',
                                     'Project 5 Cost': '1067',
                                     'Project 6 Cost': '552',
                                     'Project 7 Cost': '2991',
                                     'Project 8 Cost': '1727',
                                     'Project 9 Cost': '1639'}}],
             'returned_rows': 11,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [11, 11], "
                                   "'expected_shape': [11, 11], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [11, 11], "
                                   "'expected_shape': [11, 11], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    frame = CSVQA_FRAMES['file_0_view_0']
    managers = []
    projects = []
    cost = {}
    managers = list(frame['Manager'])
    project_cost_columns = [col for col in frame.columns if col != 'Manager']
    projects = [col.replace(' Cost', '') for col in project_cost_columns]
    cost = {manager: {} for manager in managers}
    for (idx, row) in frame.iterrows():
        manager = row['Manager']
        for col in project_cost_columns:
            project = col.replace(' Cost', '')
            try:
                cost[manager][project] = float(row[col])
            except Exception:
                raise ValueError(f"Non-numeric or missing cost for manager '{manager}', project '{project}': '{row[col]}'")
    if len(managers) != len(projects):
        raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
    for manager in managers:
        for project in projects:
            if project not in cost[manager]:
                raise ValueError(f"Missing cost entry for manager '{manager}', project '{project}'.")
    m = gp.Model('Manager_Project_Assignment')
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