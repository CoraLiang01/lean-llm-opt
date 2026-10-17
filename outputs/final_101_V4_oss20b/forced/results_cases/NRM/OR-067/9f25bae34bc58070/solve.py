CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'NRM',
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
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'))
    columns = table['columns']
    records = table['records']
    manager_col = 'Unnamed: 0'
    managers = {rec['values'][manager_col] for rec in records}
    projects = [col for col in columns if col != manager_col]
    c = {}
    for rec in records:
        m = rec['values'][manager_col]
        for p in projects:
            val_str = rec['values'][p]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Non-numeric cost for manager {m}, project {p}')
            c[m, p] = val
    if len(c) != len(managers) * len(projects):
        missing = set(((m, p) for m in managers for p in projects)) - set(c.keys())
        raise ValueError(f'Missing cost entries for pairs: {missing}')
    m = gurobipy.Model('assignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m_id, p_id) for m_id in managers for p_id in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c[m_id, p_id] * x[m_id, p_id] for m_id in managers for p_id in projects)), GRB.MINIMIZE)
    for m_id in managers:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for p_id in projects)) == 1, name='')
    for p_id in projects:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for m_id in managers)) == 1, name='')
    m.optimize()
    return m
m = solve_problem()