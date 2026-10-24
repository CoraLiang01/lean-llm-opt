CSVQA_DATA = {'query': 'Now we need to determine 10 out of 12 workers to each complete one of the ten tasks. Since each worker has different skill sets, the amount of time required for each worker to complete each task is also different. The time required for each worker to complete each task is shown in 15.csv.\n\n    Try to find a work assignment plan that minimizes the total working hours.', 'relationships': [], 'tables': [{'columns': ['Task Time Required', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 'file_name': '15.csv', 'records': [{'data_row_number': 0, 'values': {'A': '', 'B': '', 'C': '', 'D': '', 'E': '', 'F': '', 'G': '', 'H': '', 'I': '', 'J': '', 'Task Time Required': 'Worker'}}, {'data_row_number': 1, 'values': {'A': '9', 'B': '4', 'C': '3', 'D': '7', 'E': '6', 'F': '5', 'G': '6', 'H': '3', 'I': '7', 'J': '5', 'Task Time Required': '1'}}, {'data_row_number': 2, 'values': {'A': '4', 'B': '6', 'C': '5', 'D': '6', 'E': '4', 'F': '5', 'G': '3', 'H': '8', 'I': '7', 'J': '6', 'Task Time Required': '2'}}, {'data_row_number': 3, 'values': {'A': '5', 'B': '4', 'C': '7', 'D': '5', 'E': '6', 'F': '6', 'G': '5', 'H': '8', 'I': '6', 'J': '9', 'Task Time Required': '3'}}, {'data_row_number': 4, 'values': {'A': '7', 'B': '5', 'C': '2', 'D': '3', 'E': '7', 'F': '8', 'G': '5', 'H': '6', 'I': '8', 'J': '5', 'Task Time Required': '4'}}, {'data_row_number': 5, 'values': {'A': '10', 'B': '6', 'C': '7', 'D': '4', 'E': '5', 'F': '4', 'G': '4', 'H': '5', 'I': '9', 'J': '7', 'Task Time Required': '5'}}, {'data_row_number': 6, 'values': {'A': '6', 'B': '7', 'C': '6', 'D': '3', 'E': '9', 'F': '5', 'G': '7', 'H': '4', 'I': '3', 'J': '4', 'Task Time Required': '6'}}, {'data_row_number': 7, 'values': {'A': '8', 'B': '8', 'C': '5', 'D': '9', 'E': '5', 'F': '7', 'G': '5', 'H': '9', 'I': '5', 'J': '3', 'Task Time Required': '7'}}, {'data_row_number': 8, 'values': {'A': '7', 'B': '4', 'C': '8', 'D': '8', 'E': '6', 'F': '7', 'G': '5', 'H': '7', 'I': '7', 'J': '7', 'Task Time Required': '8'}}, {'data_row_number': 9, 'values': {'A': '5', 'B': '6', 'C': '8', 'D': '7', 'E': '7', 'F': '8', 'G': '7', 'H': '8', 'I': '4', 'J': '5', 'Task Time Required': '9'}}, {'data_row_number': 10, 'values': {'A': '8', 'B': '7', 'C': '9', 'D': '5', 'E': '8', 'F': '5', 'G': '9', 'H': '9', 'I': '3', 'J': '4', 'Task Time Required': '10'}}, {'data_row_number': 11, 'values': {'A': '9', 'B': '8', 'C': '10', 'D': '8', 'E': '5', 'F': '4', 'G': '7', 'H': '6', 'I': '8', 'J': '7', 'Task Time Required': '11'}}, {'data_row_number': 12, 'values': {'A': '8', 'B': '5', 'C': '6', 'D': '9', 'E': '4', 'F': '7', 'G': '8', 'H': '4', 'I': '7', 'J': '9', 'Task Time Required': '12'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f'Table with id {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the table')
    header = records[0]
    task_columns = [col for col in table.get('columns', []) if col != 'Task Time Required']
    cost = {}
    for rec in records[1:]:
        values = rec.get('values', {})
        worker_id = values.get('Task Time Required')
        if worker_id is None or worker_id == '':
            raise ValueError(f'Missing worker identifier in record {rec}')
        worker_id = str(worker_id).strip()
        cost[worker_id] = {}
        for t in task_columns:
            val = values.get(t)
            if val is None or val == '':
                raise ValueError(f'Missing cost for worker {worker_id}, task {t}')
            try:
                cost[worker_id][t] = float(val)
            except ValueError:
                raise ValueError(f'Non-numeric cost for worker {worker_id}, task {t}: {val}')
    workers = list(cost.keys())
    tasks = task_columns
    for w in workers:
        if set(cost[w].keys()) != set(tasks):
            missing = set(tasks) - set(cost[w].keys())
            raise ValueError(f'Worker {w} missing tasks: {missing}')
    m = gurobipy.Model('assignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, t) for i in workers for t in tasks], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[i][t] * x[i, t] for i in workers for t in tasks)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, t] for i in workers)) == 1 for t in tasks), name='')
    m.addConstrs((gurobipy.quicksum((x[i, t] for t in tasks)) <= 1 for i in workers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
