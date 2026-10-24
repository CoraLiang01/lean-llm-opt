CSVQA_DATA = {'query': 'In the construction industry, there are multiple managers and construction projects, with each manager incurring different costs for different projects based on their expertise and experience. This cost data is recorded in the “manager_project_costs.csv” file, where each row represents a manager and each column indicates the cost for that manager to complete a specific project. The objective is to determine the most cost-effective assignment of managers to projects, ensuring that each project is assigned to a single manager and each manager is responsible for only one project. The goal is to minimize the total cost while satisfying these assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Manager', 'Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Manager': 'Manager 1', 'Project 1 Cost': '2972', 'Project 2 Cost': '2727', 'Project 3 Cost': '2795', 'Project 4 Cost': '2922', 'Project 5 Cost': '1302', 'Project 6 Cost': '2489', 'Project 7 Cost': '1533'}}, {'data_row_number': 1, 'values': {'Manager': 'Manager 2', 'Project 1 Cost': '1094', 'Project 2 Cost': '2158', 'Project 3 Cost': '2990', 'Project 4 Cost': '1844', 'Project 5 Cost': '2887', 'Project 6 Cost': '2021', 'Project 7 Cost': '2288'}}, {'data_row_number': 2, 'values': {'Manager': 'Manager 3', 'Project 1 Cost': '2133', 'Project 2 Cost': '1675', 'Project 3 Cost': '2422', 'Project 4 Cost': '2639', 'Project 5 Cost': '1033', 'Project 6 Cost': '2261', 'Project 7 Cost': '1695'}}, {'data_row_number': 3, 'values': {'Manager': 'Manager 4', 'Project 1 Cost': '1951', 'Project 2 Cost': '2309', 'Project 3 Cost': '2070', 'Project 4 Cost': '2802', 'Project 5 Cost': '2328', 'Project 6 Cost': '1313', 'Project 7 Cost': '2434'}}, {'data_row_number': 4, 'values': {'Manager': 'Manager 5', 'Project 1 Cost': '1269', 'Project 2 Cost': '2153', 'Project 3 Cost': '1296', 'Project 4 Cost': '2685', 'Project 5 Cost': '2627', 'Project 6 Cost': '1610', 'Project 7 Cost': '1641'}}, {'data_row_number': 5, 'values': {'Manager': 'Manager 6', 'Project 1 Cost': '1220', 'Project 2 Cost': '1192', 'Project 3 Cost': '2907', 'Project 4 Cost': '2622', 'Project 5 Cost': '2595', 'Project 6 Cost': '1261', 'Project 7 Cost': '2384'}}, {'data_row_number': 6, 'values': {'Manager': 'Manager 7', 'Project 1 Cost': '1286', 'Project 2 Cost': '1659', 'Project 3 Cost': '1179', 'Project 4 Cost': '1348', 'Project 5 Cost': '1420', 'Project 6 Cost': '2862', 'Project 7 Cost': '1959'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    managers = []
    project_columns = []
    for rec in table['records']:
        values = rec['values']
        manager = values['Manager']
        if manager not in managers:
            managers.append(manager)
        for col in values:
            if col.endswith('Cost'):
                proj_col = col
                if proj_col not in project_columns:
                    project_columns.append(proj_col)
    projects = []
    for col in project_columns:
        parts = col.split()
        if len(parts) >= 3 and parts[0] == 'Project' and (parts[2] == 'Cost'):
            proj_id = f'{parts[0]} {parts[1]}'
            projects.append(proj_id)
        else:
            raise ValueError(f'Unexpected project column format: {col}')
    c = {}
    for rec in table['records']:
        values = rec['values']
        manager = values['Manager']
        for col in project_columns:
            proj_id = f'{col.split()[0]} {col.split()[1]}'
            if proj_id not in projects:
                continue
            key = (manager, proj_id)
            try:
                cost_val = float(values[col])
            except (KeyError, ValueError):
                raise ValueError(f'Missing or invalid cost for {manager} on {proj_id}')
            c[key] = cost_val
    for i in managers:
        for j in projects:
            if (i, j) not in c:
                raise ValueError(f'Missing cost for manager {i} and project {j}')
    m = gurobipy.Model('assignment_problem')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in managers for j in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c[i, j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
