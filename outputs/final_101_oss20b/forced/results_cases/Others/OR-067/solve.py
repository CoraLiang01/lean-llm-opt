CSVQA_DATA = {'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager incurs different costs for each project, based on their experience and expertise. The cost information for each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a manager, and each column represents the cost for that manager to complete a specific project. The objective is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal is to minimize the total cost while satisfying these one-to-one assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    M = [rec['values']['Unnamed: 0'] for rec in records]
    P = [col for col in table.get('columns', []) if col != 'Unnamed: 0']
    c = {}
    for rec in records:
        m = rec['values']['Unnamed: 0']
        for p in P:
            val = rec['values'].get(p)
            if val is None:
                raise ValueError(f'Missing cost for manager {m} and project {p}')
            try:
                c[m, p] = float(val)
            except ValueError as e:
                raise ValueError(f'Invalid numeric value for cost[{m},{p}]: {val}') from e
    for m in M:
        for p in P:
            if (m, p) not in c:
                raise ValueError(f'Cost entry missing for manager {m} and project {p}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(m, p) for m in M for p in P], vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((c[m, p] * x[m, p] for m in M for p in P)), GRB.MINIMIZE)
    for m in M:
        model.addConstr(gurobipy.quicksum((x[m, p] for p in P)) == 1, name=f'Manager_{m}')
    for p in P:
        model.addConstr(gurobipy.quicksum((x[m, p] for m in M)) == 1, name=f'Project_{p}')
    model.optimize()
    return model
m = solve_problem()
