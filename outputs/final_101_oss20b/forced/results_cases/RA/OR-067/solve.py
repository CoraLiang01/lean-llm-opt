CSVQA_DATA = {'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager incurs different costs for each project, based on their experience and expertise. The cost information for each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a manager, and each column represents the cost for that manager to complete a specific project. The objective is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal is to minimize the total cost while satisfying these one-to-one assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    columns = table.get('columns', [])
    if not columns:
        raise ValueError(f'No columns found in table {table_id}')
    manager_col = 'Unnamed: 0'
    if manager_col not in columns:
        raise ValueError(f"Manager column '{manager_col}' not found in table {table_id}")
    project_cols = [c for c in columns if c != manager_col]
    if not project_cols:
        raise ValueError(f'No project columns found in table {table_id}')
    records = table.get('records', [])
    managers = []
    for rec in records:
        val = rec.get('values', {})
        m_id = val.get(manager_col)
        if m_id is None:
            raise ValueError(f'Missing manager id in record {rec}')
        managers.append(m_id)
    projects = project_cols
    cost = {}
    for rec in records:
        vals = rec.get('values', {})
        m_id = vals.get(manager_col)
        for p in projects:
            if p not in vals:
                raise ValueError(f'Missing cost for manager {m_id} project {p}')
            try:
                c_val = float(vals[p])
            except ValueError:
                raise ValueError(f'Non-numeric cost for manager {m_id} project {p}: {vals[p]}')
            cost[m_id, p] = c_val
    for m in managers:
        for p in projects:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost entry for manager {m} and project {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    var_keys = [(m_id, p_id) for m_id in managers for p_id in projects]
    x = m.addVars(var_keys, vtype=GRB.BINARY, name='')
    obj_expr = gurobipy.quicksum((cost[m_id, p_id] * x[m_id, p_id] for m_id in managers for p_id in projects))
    m.setObjective(obj_expr, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[m_id, p_id] for p_id in projects)) == 1 for m_id in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[m_id, p_id] for m_id in managers)) == 1 for p_id in projects), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
