LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv", "values": {"Process": "BI", "1": "1.1", "2": "2.1", "3": "3", "4": "1", "5": "0.7", "6": "5", "7": "3", "8": "3.5", "9": "4.4", "10": "3.8", "11": "3.5", "12": "2.8", "13": "4.1", "14": "2.9", "15": "5.4", "16": "5.8", "17": "2.6", "18": "4.9", "19": "3.4", "20": "3.6", "21": "5.6", "22": "0.9", "23": "1", "24": "0.6", "25": "5.1", "26": "4.8", "27": "5.3", "28": "5.9", "29": "4.9", "30": "3", "31": "4.8", "32": "1.2", "33": "4", "34": "1.3", "35": "5.7", "36": "3.4", "37": "2.8", "38": "2", "39": "4.8", "40": "3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
task_record = None
for rec in LEGACY_RECORDS:
    if rec['source'] == '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv':
        task_record = rec
        break
if task_record is None:
    raise ValueError('Required task data not found in LEGACY_RECORDS.')
tasks = [str(i) for i in range(1, 41)]
processors = ['1', '2', '3']
frequencies = {'1': 1.33, '2': 2.0, '3': 2.66}
if 'values' not in task_record or not all((t in task_record['values'] for t in tasks)):
    raise ValueError('Missing task instruction counts in LEGACY_RECORDS.')
b_t = {t: float(task_record['values'][t]) for t in tasks}
m = gp.Model('TaskAssignment')
x_vars = m.addVars(tasks, processors, vtype=GRB.BINARY, lb=0, ub=1, name='')
Cmax = m.addVar(vtype=GRB.CONTINUOUS, name='Cmax', lb=0)
m.addConstrs((gp.quicksum((x_vars[t, p] for p in processors)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((b_t[t] / frequencies[p] * x_vars[t, p] for t in tasks)) <= Cmax for p in processors), name='')
m.setObjective(Cmax, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')