CSVQA_DATA = {'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager incurs different costs for each project, based on their experience and expertise. The cost information for each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a manager, and each column represents the cost for that manager to complete a specific project. The objective is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal is to minimize the total cost while satisfying these one-to-one assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'), None)
    if table is None:
        raise ValueError('Table file_0_view_0 not found')
    columns = table['columns']
    manager_col = columns[0]
    project_cols = columns[1:]
    M = set()
    P = set(project_cols)
    cost = {}
    for rec in table['records']:
        values = rec['values']
        mgr = values[manager_col]
        M.add(mgr)
        for proj in project_cols:
            key = (mgr, proj)
            val_str = values[proj]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid cost value for manager {mgr}, project {proj}: {val_str}')
            cost[key] = val
    for mgr in M:
        for proj in P:
            if (mgr, proj) not in cost:
                raise ValueError(f'Missing cost for manager {mgr}, project {proj}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    var_keys = [(mgr, proj) for mgr in M for proj in P]
    x = model.addVars(var_keys, vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((cost[mgr, proj] * x[mgr, proj] for mgr in M for proj in P)), GRB.MINIMIZE)
    model.addConstrs((model.addConstr(gurobipy.quicksum((x[mgr, proj] for proj in P)) == 1, name='') for mgr in M), name='')
    model.addConstrs((model.addConstr(gurobipy.quicksum((x[mgr, proj] for mgr in M)) == 1, name='') for proj in P), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
