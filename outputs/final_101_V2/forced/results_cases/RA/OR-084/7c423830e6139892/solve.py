LEGACY_OBSERVATION = 'Process,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40\nBI,1.1,2.1,3,1,0.7,5,3,3.5,4.4,3.8,3.5,2.8,4.1,2.9,5.4,5.8,2.6,4.9,3.4,3.6,5.6,0.9,1,0.6,5.1,4.8,5.3,5.9,4.9,3,4.8,1.2,4,1.3,5.7,3.4,2.8,2,4.8,3'
LEGACY_RECORDS = [{'source': '', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
record = LEGACY_RECORDS[0]
if record['source'] == '' and 'Process' in record['values'] and (record['values']['Process'] == 'BI'):
    tasks = [str(i) for i in range(1, 41)]
    b = {i: float(record['values'][i]) for i in tasks}
else:
    raise ValueError('BI data not found in LEGACY_RECORDS')
cpus = ['1', '2', '3']
freq = {'1': 1.33, '2': 2.0, '3': 2.66}
m = gp.Model('TaskAssignment')
x = m.addVars(tasks, cpus, vtype=GRB.BINARY, name='')
Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Cmax')
m.addConstrs((gp.quicksum((x[i, j] for j in cpus)) == 1 for i in tasks), name='')
m.addConstrs((gp.quicksum((b[i] / freq[j] * x[i, j] for i in tasks)) <= Cmax for j in cpus), name='')
m.setObjective(Cmax, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
if set(tasks) != set(b.keys()):
    raise ValueError('Mismatch between task indices and BI data')
if set(cpus) != set(freq.keys()):
    raise ValueError('Mismatch between CPU indices and frequency data')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')