CSVQA_DATA = {'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager incurs different costs for each project, based on their experience and expertise. The cost information for each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a manager, and each column represents the cost for that manager to complete a specific project. The objective is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal is to minimize the total cost while satisfying these one-to-one assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    columns = table['columns']
    manager_col = 'Unnamed: 0'
    project_cols = [c for c in columns if c != manager_col]
    managers = [rec['values'][manager_col] for rec in records]
    projects = project_cols
    cost = {}
    for rec in records:
        i = rec['values'][manager_col]
        for j in project_cols:
            if j not in rec['values']:
                raise ValueError(f'Missing cost for manager {i} and project {j}')
            try:
                val = float(rec['values'][j])
            except ValueError:
                raise ValueError(f'Non-numeric cost for manager {i} and project {j}')
            if val < 0:
                raise ValueError(f'Negative cost for manager {i} and project {j}')
            cost[i, j] = val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in managers for j in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
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
