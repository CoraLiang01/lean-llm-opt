LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv","values":{"Process":"BI","1":"1.1","2":"2.1","3":"3","4":"1","5":"0.7","6":"5","7":"3","8":"3.5","9":"4.4","10":"3.8","11":"3.5","12":"2.8","13":"4.1","14":"2.9","15":"5.4","16":"5.8","17":"2.6","18":"4.9","19":"3.4","20":"3.6","21":"5.6","22":"0.9","23":"1","24":"0.6","25":"5.1","26":"4.8","27":"5.3","28":"5.9","29":"4.9","30":"3","31":"4.8","32":"1.2","33":"4","34":"1.3","35":"5.7","36":"3.4","37":"2.8","38":"2","39":"4.8","40":"3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
task_record = None
for rec in LEGACY_RECORDS:
    if rec['source'] == '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv':
        task_record = rec
        break
if task_record is None:
    raise ValueError('Task instruction counts not found in LEGACY_RECORDS.')
tasks = [str(i) for i in range(1, 41)]
cpus = [str(i) for i in range(1, 4)]
b = {}
for t in tasks:
    if t not in task_record['values']:
        raise ValueError(f'Missing instruction count for task {t}.')
    b[t] = float(task_record['values'][t])
cpu_frequencies = {'1': 1.33, '2': 2.0, '3': 2.66}
for p in cpus:
    if p not in cpu_frequencies:
        raise ValueError(f'Missing frequency for CPU {p}.')
m = gp.Model('TaskAssignmentMakespan')
x_vars = m.addVars(tasks, cpus, vtype=GRB.BINARY, name='')
Cmax = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='Cmax')
m.addConstrs((gp.quicksum((x_vars[t, p] for p in cpus)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((b[t] / cpu_frequencies[p] * x_vars[t, p] for t in tasks)) <= Cmax for p in cpus), name='')
m.setObjective(Cmax, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')