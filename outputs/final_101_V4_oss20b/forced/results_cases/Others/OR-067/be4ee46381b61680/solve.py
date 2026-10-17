CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {},
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
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    M = [rec['values']['Unnamed: 0'] for rec in records]
    P = [col for col in table['columns'] if col != 'Unnamed: 0']
    cost = {}
    for rec in records:
        m = rec['values']['Unnamed: 0']
        for p in P:
            val_str = rec['values'][p]
            try:
                val = float(val_str)
            except Exception as e:
                raise ValueError(f'Invalid cost value for manager {m}, project {p}: {val_str}') from e
            cost[m, p] = val
    for m in M:
        for p in P:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost for manager {m}, project {p}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(m, p) for m in M for p in P], vtype=GRB.BINARY, name='')
    for m in M:
        model.addConstr(gurobipy.quicksum((x[m, p] for p in P)) == 1, name=f'manager_{m}')
    for p in P:
        model.addConstr(gurobipy.quicksum((x[m, p] for m in M)) == 1, name=f'project_{p}')
    obj = gurobipy.quicksum((cost[m, p] * x[m, p] for m in M for p in P))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    return model
m = solve_problem()