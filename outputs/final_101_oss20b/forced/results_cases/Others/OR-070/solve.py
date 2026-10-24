CSVQA_DATA = {'query': 'In the construction industry, there are multiple managers and construction projects, with each manager incurring different costs for different projects based on their expertise and experience. This cost data is recorded in the “manager_project_costs.csv” file, where each row represents a manager and each column indicates the cost for that manager to complete a specific project. The objective is to determine the most cost-effective assignment of managers to projects, ensuring that each project is assigned to a single manager and each manager is responsible for only one project. The goal is to minimize the total cost while satisfying these assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Manager', 'Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Manager': 'Manager 1', 'Project 1 Cost': '2972', 'Project 2 Cost': '2727', 'Project 3 Cost': '2795', 'Project 4 Cost': '2922', 'Project 5 Cost': '1302', 'Project 6 Cost': '2489', 'Project 7 Cost': '1533'}}, {'data_row_number': 1, 'values': {'Manager': 'Manager 2', 'Project 1 Cost': '1094', 'Project 2 Cost': '2158', 'Project 3 Cost': '2990', 'Project 4 Cost': '1844', 'Project 5 Cost': '2887', 'Project 6 Cost': '2021', 'Project 7 Cost': '2288'}}, {'data_row_number': 2, 'values': {'Manager': 'Manager 3', 'Project 1 Cost': '2133', 'Project 2 Cost': '1675', 'Project 3 Cost': '2422', 'Project 4 Cost': '2639', 'Project 5 Cost': '1033', 'Project 6 Cost': '2261', 'Project 7 Cost': '1695'}}, {'data_row_number': 3, 'values': {'Manager': 'Manager 4', 'Project 1 Cost': '1951', 'Project 2 Cost': '2309', 'Project 3 Cost': '2070', 'Project 4 Cost': '2802', 'Project 5 Cost': '2328', 'Project 6 Cost': '1313', 'Project 7 Cost': '2434'}}, {'data_row_number': 4, 'values': {'Manager': 'Manager 5', 'Project 1 Cost': '1269', 'Project 2 Cost': '2153', 'Project 3 Cost': '1296', 'Project 4 Cost': '2685', 'Project 5 Cost': '2627', 'Project 6 Cost': '1610', 'Project 7 Cost': '1641'}}, {'data_row_number': 5, 'values': {'Manager': 'Manager 6', 'Project 1 Cost': '1220', 'Project 2 Cost': '1192', 'Project 3 Cost': '2907', 'Project 4 Cost': '2622', 'Project 5 Cost': '2595', 'Project 6 Cost': '1261', 'Project 7 Cost': '2384'}}, {'data_row_number': 6, 'values': {'Manager': 'Manager 7', 'Project 1 Cost': '1286', 'Project 2 Cost': '1659', 'Project 3 Cost': '1179', 'Project 4 Cost': '1348', 'Project 5 Cost': '1420', 'Project 6 Cost': '2862', 'Project 7 Cost': '1959'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    data = CSVQA_DATA
    table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    managers = [rec['values']['Manager'] for rec in records]
    proj_cols = [col for col in table['columns'] if col.startswith('Project ') and col.endswith(' Cost')]
    projects = [int(re.search('Project (\\d+) Cost', col).group(1)) for col in proj_cols]
    c = {}
    for rec in records:
        m_id = rec['values']['Manager']
        for p, col in zip(projects, proj_cols):
            val_str = rec['values'][col]
            if val_str is None:
                raise ValueError(f'Missing cost for manager {m_id} project {p}')
            c[m_id, p] = int(val_str)
    for m_id in managers:
        for p in projects:
            if (m_id, p) not in c:
                raise ValueError(f'Missing cost for manager {m_id} project {p}')
    model = gurobipy.Model('Assignment')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(m_id, p) for m_id in managers for p in projects], vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((c[i, j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    for m_id in managers:
        model.addConstr(gurobipy.quicksum((x[m_id, p] for p in projects)) <= 1, name=f'Mgr_{m_id}_max')
    for p in projects:
        model.addConstr(gurobipy.quicksum((x[m_id, p] for m_id in managers)) == 1, name=f'Proj_{p}_exact')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
