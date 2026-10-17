CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'RA',
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
    data = CSVQA_DATA
    table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    columns = table['columns']
    M = set((record['values']['Unnamed: 0'] for record in records))
    P = [col for col in columns if col != 'Unnamed: 0']
    cost = {}
    for record in records:
        m = record['values']['Unnamed: 0']
        for p in P:
            val_str = record['values'][p]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Non-numeric cost value for manager {m}, project {p}: {val_str}')
            if val < 0:
                raise ValueError(f'Negative cost for manager {m}, project {p}: {val}')
            cost[m, p] = val
    for m in M:
        for p in P:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost for manager {m}, project {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m_id, p_id) for m_id in M for p_id in P], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[m_id, p_id] * x[m_id, p_id] for m_id in M for p_id in P)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[m_id, p_id] for p_id in P)) == 1 for m_id in M), name='')
    m.addConstrs((gurobipy.quicksum((x[m_id, p_id] for m_id in M)) == 1 for p_id in P), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()