CSVQA_DATA = {'query': 'In the construction industry, a set of managers and a set of construction projects are given. Assigning manager i to project j incurs a cost that depends on the manager’s experience and expertise. These costs are provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each column corresponds to a project. The task is to determine a minimum-cost one-to-one assignment of managers to projects, such that each manager is assigned to exactly one project and each project is assigned to exactly one manager.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '2216', 'P2': '1911', 'P3': '1661', 'P4': '2122', 'P5': '1442', 'P6': '1442', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '1100', 'P2': '1271', 'P3': '2764', 'P4': '2557', 'P5': '1036', 'P6': '1036', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2827', 'P2': '2784', 'P3': '2206', 'P4': '2216', 'P5': '2677', 'P6': '2677', 'Unnamed: 0': 'MC'}}, {'data_row_number': 3, 'values': {'P1': '2627', 'P2': '1273', 'P3': '2610', 'P4': '1957', 'P5': '1594', 'P6': '1594', 'Unnamed: 0': 'MD'}}, {'data_row_number': 4, 'values': {'P1': '3359', 'P2': '1003', 'P3': '2554', 'P4': '1706', 'P5': '2065', 'P6': '2065', 'Unnamed: 0': 'ME'}}, {'data_row_number': 5, 'values': {'P1': '1579', 'P2': '2289', 'P3': '2368', 'P4': '1922', 'P5': '2740', 'P6': '2740', 'Unnamed: 0': 'MF'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = None
    for t in CSVQA_DATA['tables']:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found')
    cols = table['columns']
    projects = [c for c in cols if c != 'Unnamed: 0']
    managers = []
    cost = {}
    for rec in table['records']:
        values = rec['values']
        manager = values['Unnamed: 0']
        managers.append(manager)
        cost[manager] = {}
        for proj in projects:
            val_str = values.get(proj)
            if val_str is None:
                raise ValueError(f'Missing cost for manager {manager} project {proj}')
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid numeric cost for manager {manager} project {proj}: {val_str}')
            cost[manager][proj] = val
    for mng in managers:
        if mng not in cost:
            raise ValueError(f'Missing cost data for manager {mng}')
        for proj in projects:
            if proj not in cost[mng]:
                raise ValueError(f'Missing cost for manager {mng} project {proj}')
    x = m.addVars([(i, j) for i in managers for j in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
    m.optimize()
    return m
m = solve_problem()
