CSVQA_DATA = {'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of the last task.', 'relationships': [], 'tables': [{'columns': ['Process', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'], 'file_name': '18.csv', 'records': [{'data_row_number': 0, 'values': {'1': '1.1', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '2': '2.1', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '3': '3', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '4': '1', '40': '3', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', 'Process': 'BI'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError(f'No records found in table {table_id}')
    record = records[0]
    values = record.get('values', {})
    I = [str(i) for i in range(1, 41)]
    J = [1, 2, 3]
    s_j = {1: 1.33, 2: 2.0, 3: 2.66}
    p_i = {}
    for i in I:
        if i not in values:
            raise ValueError(f'Missing instruction value for task {i}')
        try:
            p_i[i] = float(values[i])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for task {i}: {values[i]}') from e
    model = gurobipy.Model('TaskAssignment')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    Cmax = model.addVar(lb=0, name='Cmax')
    model.setObjective(Cmax, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) == 1 for i in I), name='')
    for j in J:
        load_expr = gurobipy.quicksum((p_i[i] / s_j[j] * x[i, j] for i in I))
        model.addConstr(load_expr <= Cmax, name=f'cpu_load_{j}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
