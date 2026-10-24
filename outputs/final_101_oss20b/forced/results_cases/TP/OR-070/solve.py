CSVQA_DATA = {'query': 'In the construction industry, there are multiple managers and construction projects, with each manager incurring different costs for different projects based on their expertise and experience. This cost data is recorded in the “manager_project_costs.csv” file, where each row represents a manager and each column indicates the cost for that manager to complete a specific project. The objective is to determine the most cost-effective assignment of managers to projects, ensuring that each project is assigned to a single manager and each manager is responsible for only one project. The goal is to minimize the total cost while satisfying these assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Manager', 'Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Manager': 'Manager 1', 'Project 1 Cost': '2972', 'Project 2 Cost': '2727', 'Project 3 Cost': '2795', 'Project 4 Cost': '2922', 'Project 5 Cost': '1302', 'Project 6 Cost': '2489', 'Project 7 Cost': '1533'}}, {'data_row_number': 1, 'values': {'Manager': 'Manager 2', 'Project 1 Cost': '1094', 'Project 2 Cost': '2158', 'Project 3 Cost': '2990', 'Project 4 Cost': '1844', 'Project 5 Cost': '2887', 'Project 6 Cost': '2021', 'Project 7 Cost': '2288'}}, {'data_row_number': 2, 'values': {'Manager': 'Manager 3', 'Project 1 Cost': '2133', 'Project 2 Cost': '1675', 'Project 3 Cost': '2422', 'Project 4 Cost': '2639', 'Project 5 Cost': '1033', 'Project 6 Cost': '2261', 'Project 7 Cost': '1695'}}, {'data_row_number': 3, 'values': {'Manager': 'Manager 4', 'Project 1 Cost': '1951', 'Project 2 Cost': '2309', 'Project 3 Cost': '2070', 'Project 4 Cost': '2802', 'Project 5 Cost': '2328', 'Project 6 Cost': '1313', 'Project 7 Cost': '2434'}}, {'data_row_number': 4, 'values': {'Manager': 'Manager 5', 'Project 1 Cost': '1269', 'Project 2 Cost': '2153', 'Project 3 Cost': '1296', 'Project 4 Cost': '2685', 'Project 5 Cost': '2627', 'Project 6 Cost': '1610', 'Project 7 Cost': '1641'}}, {'data_row_number': 5, 'values': {'Manager': 'Manager 6', 'Project 1 Cost': '1220', 'Project 2 Cost': '1192', 'Project 3 Cost': '2907', 'Project 4 Cost': '2622', 'Project 5 Cost': '2595', 'Project 6 Cost': '1261', 'Project 7 Cost': '2384'}}, {'data_row_number': 6, 'values': {'Manager': 'Manager 7', 'Project 1 Cost': '1286', 'Project 2 Cost': '1659', 'Project 3 Cost': '1179', 'Project 4 Cost': '1348', 'Project 5 Cost': '1420', 'Project 6 Cost': '2862', 'Project 7 Cost': '1959'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    managers = [rec['values']['Manager'] for rec in table['records']]
    projects = [col for col in table['columns'] if col != 'Manager']
    cost = {m: {} for m in managers}
    for rec in table['records']:
        m = rec['values']['Manager']
        for p in projects:
            val = rec['values'][p]
            try:
                cost[m][p] = float(val)
            except (TypeError, ValueError):
                raise ValueError(f'Invalid cost value for manager {m}, project {p}: {val}')
    for m in managers:
        if m not in cost:
            raise ValueError(f'Missing cost data for manager {m}')
        for p in projects:
            if p not in cost[m]:
                raise ValueError(f'Missing cost data for manager {m}, project {p}')
    m = gurobipy.Model('manager_project_assignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m_id, p_id) for m_id in managers for p_id in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[m_id][p_id] * x[m_id, p_id] for m_id in managers for p_id in projects)), GRB.MINIMIZE)
    for m_id in managers:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for p_id in projects)) == 1, name=f'manager_{m_id}_assignment')
    for p_id in projects:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for m_id in managers)) == 1, name=f'project_{p_id}_assignment')
    m.optimize()
    return m
m = solve_problem()
