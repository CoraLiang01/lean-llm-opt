CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'AP',
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
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    columns = target_table.get('columns', [])
    if not columns or len(columns) < 2:
        raise ValueError('Table must have at least one manager column and one project column')
    manager_col = columns[0]
    project_cols = columns[1:]
    M = set()
    P = set(project_cols)
    c = {}
    for record in target_table.get('records', []):
        values = record.get('values', {})
        m_id = values.get(manager_col)
        if m_id is None:
            raise ValueError(f'Missing manager identifier in record {record}')
        M.add(m_id)
        for p in project_cols:
            val_str = values.get(p)
            if val_str is None:
                raise ValueError(f'Missing cost for manager {m_id} and project {p}')
            try:
                cost_val = float(val_str)
            except ValueError:
                raise ValueError(f"Non-numeric cost '{val_str}' for manager {m_id}, project {p}")
            c[m_id, p] = cost_val
    for m in M:
        for p in P:
            if (m, p) not in c:
                raise ValueError(f'Missing cost entry for manager {m} and project {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m_id, p_id) for m_id in M for p_id in P], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c[m_id, p_id] * x[m_id, p_id] for m_id in M for p_id in P)), GRB.MINIMIZE)
    for m_id in M:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for p_id in P)) == 1)
    for p_id in P:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for m_id in M)) == 1)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()