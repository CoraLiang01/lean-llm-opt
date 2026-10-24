CSVQA_DATA = {'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager incurs different costs for each project, based on their experience and expertise. The cost information for each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a manager, and each column represents the cost for that manager to complete a specific project. The objective is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal is to minimize the total cost while satisfying these one-to-one assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == table_id:
            table = tbl
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    columns = table['columns']
    if not columns:
        raise ValueError('No columns found in the table')
    manager_col = columns[0]
    project_cols = columns[1:]
    managers = []
    for rec in table['records']:
        mgr = rec['values'].get(manager_col)
        if mgr is None:
            raise ValueError(f'Missing manager identifier in record {rec}')
        managers.append(mgr)
    cost = {}
    for rec in table['records']:
        mgr = rec['values'][manager_col]
        cost[mgr] = {}
        for proj in project_cols:
            val = rec['values'].get(proj)
            if val is None:
                raise ValueError(f'Missing cost for manager {mgr} and project {proj}')
            try:
                cost[mgr][proj] = float(val)
            except ValueError:
                raise ValueError(f"Non-numeric cost value '{val}' for manager {mgr}, project {proj}")
    for mgr in managers:
        if set(cost[mgr].keys()) != set(project_cols):
            missing = set(project_cols) - set(cost[mgr].keys())
            raise ValueError(f'Manager {mgr} missing costs for projects: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    var_keys = [(mgr, proj) for mgr in managers for proj in project_cols]
    x = m.addVars(var_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[mgr][proj] * x[mgr, proj] for mgr in managers for proj in project_cols)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[mgr, proj] for proj in project_cols)) == 1 for mgr in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[mgr, proj] for mgr in managers)) == 1 for proj in project_cols), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
