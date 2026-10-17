CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}},
                         {'source_row': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}},
                         {'source_row': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    columns = table['columns']
    manager_col = 'Unnamed: 0'
    project_cols = [c for c in columns if c != manager_col]
    I = [rec['values'][manager_col] for rec in records]
    J = project_cols
    cost = {}
    for rec in records:
        i = rec['values'][manager_col]
        for j in project_cols:
            val_str = rec['values'][j]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid cost value for manager {i}, project {j}: {val_str}')
            cost[i, j] = val
    for i in I:
        for j in J:
            if (i, j) not in cost:
                raise ValueError(f'Missing cost for manager {i} and project {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) == 1, name=f'one_proj_per_manager_{i}')
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == 1, name=f'one_mgr_per_proj_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()