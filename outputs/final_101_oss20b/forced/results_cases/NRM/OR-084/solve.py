CSVQA_DATA = {'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of the last task.', 'relationships': [], 'tables': [{'columns': ['Process', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'], 'file_name': '18.csv', 'records': [{'data_row_number': 0, 'values': {'1': '1.1', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '2': '2.1', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '3': '3', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '4': '1', '40': '3', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', 'Process': 'BI'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    bi_row = None
    for rec in table.get('records', []):
        if rec['values'].get('Process') == 'BI':
            bi_row = rec['values']
            break
    if bi_row is None:
        raise ValueError('BI row not found in table')
    b_t = {}
    for col in table['columns']:
        if col == 'Process':
            continue
        val = bi_row.get(col)
        if val is None:
            raise ValueError(f'Missing value for column {col}')
        try:
            b_t[int(col)] = float(val)
        except ValueError:
            raise ValueError(f'Non-numeric value for column {col}: {val}')
    if set(b_t.keys()) != set(range(1, 41)):
        missing = set(range(1, 41)) - set(b_t.keys())
        raise ValueError(f'Missing task data for tasks: {missing}')
    T = list(range(1, 41))
    P = [1, 2, 3]
    f_p = {1: 1.33, 2: 2.0, 3: 2.66}
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(t, p) for t in T for p in P], vtype=GRB.BINARY, name='')
    Cmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Cmax')
    model.setObjective(Cmax, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[t, p] for p in P)) == 1 for t in T), name='')
    model.addConstrs((gurobipy.quicksum((b_t[t] / f_p[p] * x[t, p] for t in T)) <= Cmax for p in P), name='')
    model.optimize()
    return model
m = solve_problem()
