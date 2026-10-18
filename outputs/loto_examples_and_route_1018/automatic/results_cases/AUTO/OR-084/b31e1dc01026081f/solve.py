LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv",\n    "values": {\n      "Process": "BI",\n      "1": "1.1",\n      "2": "2.1",\n      "3": "3",\n      "4": "1",\n      "5": "0.7",\n      "6": "5",\n      "7": "3",\n      "8": "3.5",\n      "9": "4.4",\n      "10": "3.8",\n      "11": "3.5",\n      "12": "2.8",\n      "13": "4.1",\n      "14": "2.9",\n      "15": "5.4",\n      "16": "5.8",\n      "17": "2.6",\n      "18": "4.9",\n      "19": "3.4",\n      "20": "3.6",\n      "21": "5.6",\n      "22": "0.9",\n      "23": "1",\n      "24": "0.6",\n      "25": "5.1",\n      "26": "4.8",\n      "27": "5.3",\n      "28": "5.9",\n      "29": "4.9",\n      "30": "3",\n      "31": "4.8",\n      "32": "1.2",\n      "33": "4",\n      "34": "1.3",\n      "35": "5.7",\n      "36": "3.4",\n      "37": "2.8",\n      "38": "2",\n      "39": "4.8",\n      "40": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
task_record = None
for rec in records:
    if rec['source'] and rec['source'].endswith('18.csv'):
        task_record = rec
        break
if not task_record:
    raise ValueError('Task instruction counts not found in LEGACY_RECORDS.')
tasks = [str(i) for i in range(1, 41)]
cpus = ['1', '2', '3']
cpu_freq = {'1': 1.33, '2': 2.0, '3': 2.66}
b = {}
for t in tasks:
    if t not in task_record['values']:
        raise ValueError(f'Missing instruction count for task {t}')
    b[t] = float(task_record['values'][t])
t = {}
for i in tasks:
    t[i] = {}
    for j in cpus:
        t[i][j] = b[i] / cpu_freq[j]
m = gp.Model('TaskAssignment_Makespan')
x_vars = m.addVars(tasks, cpus, vtype=GRB.BINARY, name='')
Cmax = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='Cmax')
m.setObjective(Cmax, GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in cpus)) == 1 for i in tasks), name='')
m.addConstrs((gp.quicksum((t[i][j] * x_vars[i, j] for i in tasks)) <= Cmax for j in cpus), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')