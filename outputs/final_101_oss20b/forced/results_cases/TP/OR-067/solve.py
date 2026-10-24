CSVQA_DATA = {'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager incurs different costs for each project, based on their experience and expertise. The cost information for each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a manager, and each column represents the cost for that manager to complete a specific project. The objective is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal is to minimize the total cost while satisfying these one-to-one assignment constraints.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}}, {'data_row_number': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}}, {'data_row_number': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA.get('tables', [])
table = None
for t in tables:
    if t.get('table_id') == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
records = table.get('records', [])
columns = table.get('columns', [])
I = sorted({record['values']['Unnamed: 0'] for record in records})
J = [col for col in columns if col != 'Unnamed: 0']
cost = {}
for record in records:
    i = record['values']['Unnamed: 0']
    for j in J:
        val_str = record['values'][j]
        try:
            val = float(val_str)
        except Exception as e:
            raise ValueError(f'Non-numeric cost for manager {i}, project {j}: {val_str}') from e
        cost[i, j] = val
for i in I:
    for j in J:
        if (i, j) not in cost:
            raise ValueError(f'Missing cost for manager {i}, project {j}')

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) == 1 for i in I), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) == 1 for j in J), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
