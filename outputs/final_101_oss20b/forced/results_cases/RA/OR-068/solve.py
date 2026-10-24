CSVQA_DATA = {'query': 'In the construction industry, a set of managers and a set of construction projects are given. Assigning manager i to project j incurs a cost that depends on the manager’s experience and expertise. These costs are provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each column corresponds to a project. The task is to determine a minimum-cost one-to-one assignment of managers to projects, such that each manager is assigned to exactly one project and each project is assigned to exactly one manager.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '2216', 'P2': '1911', 'P3': '1661', 'P4': '2122', 'P5': '1442', 'P6': '1442', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '1100', 'P2': '1271', 'P3': '2764', 'P4': '2557', 'P5': '1036', 'P6': '1036', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2827', 'P2': '2784', 'P3': '2206', 'P4': '2216', 'P5': '2677', 'P6': '2677', 'Unnamed: 0': 'MC'}}, {'data_row_number': 3, 'values': {'P1': '2627', 'P2': '1273', 'P3': '2610', 'P4': '1957', 'P5': '1594', 'P6': '1594', 'Unnamed: 0': 'MD'}}, {'data_row_number': 4, 'values': {'P1': '3359', 'P2': '1003', 'P3': '2554', 'P4': '1706', 'P5': '2065', 'P6': '2065', 'Unnamed: 0': 'ME'}}, {'data_row_number': 5, 'values': {'P1': '1579', 'P2': '2289', 'P3': '2368', 'P4': '1922', 'P5': '2740', 'P6': '2740', 'Unnamed: 0': 'MF'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table['records']
    columns = table['columns']
    managers = [rec['values']['Unnamed: 0'] for rec in records]
    projects = [col for col in columns if col != 'Unnamed: 0']
    cost = {}
    for rec in records:
        m_id = rec['values']['Unnamed: 0']
        for p in projects:
            val_str = rec['values'].get(p)
            if val_str is None:
                raise ValueError(f'Missing cost for manager {m_id} and project {p}')
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f"Invalid numeric cost '{val_str}' for manager {m_id}, project {p}")
            cost[m_id, p] = val
    for m in managers:
        for p in projects:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost entry for manager {m} and project {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m_id, p_id) for m_id in managers for p_id in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[m_id, p_id] * x[m_id, p_id] for m_id in managers for p_id in projects)), GRB.MINIMIZE)
    for m_id in managers:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for p_id in projects)) == 1)
    for p_id in projects:
        m.addConstr(gurobipy.quicksum((x[m_id, p_id] for m_id in managers)) == 1)
    m.optimize()
    return m
m = solve_problem()
