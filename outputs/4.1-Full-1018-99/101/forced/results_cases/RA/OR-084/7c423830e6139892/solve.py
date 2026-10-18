LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv", "values": {"Process": "BI", "1": "1.1", "2": "2.1", "3": "3", "4": "1", "5": "0.7", "6": "5", "7": "3", "8": "3.5", "9": "4.4", "10": "3.8", "11": "3.5", "12": "2.8", "13": "4.1", "14": "2.9", "15": "5.4", "16": "5.8", "17": "2.6", "18": "4.9", "19": "3.4", "20": "3.6", "21": "5.6", "22": "0.9", "23": "1", "24": "0.6", "25": "5.1", "26": "4.8", "27": "5.3", "28": "5.9", "29": "4.9", "30": "3", "31": "4.8", "32": "1.2", "33": "4", "34": "1.3", "35": "5.7", "36": "3.4", "37": "2.8", "38": "2", "39": "4.8", "40": "3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
record = next((r for r in LEGACY_RECORDS if r['source'].endswith('18.csv')))
b_t = {int(k): float(v) for k, v in record['values'].items() if k.isdigit()}
tasks = sorted(b_t.keys())
cpus = [1, 2, 3]
freq = {1: 1.33, 2: 2.0, 3: 2.66}
proc = {(t, p): b_t[t] / freq[p] for t in tasks for p in cpus}
M = sum((b_t[t] / min(freq.values()) for t in tasks))
m = gp.Model('TaskAssignment')
x_vars = m.addVars(tasks, cpus, vtype=GRB.BINARY, name='')
y_vars = m.addVars(((t, t2, p) for t in tasks for t2 in tasks if t != t2 for p in cpus), vtype=GRB.BINARY, name='')
S_vars = m.addVars(tasks, lb=0.0, vtype=GRB.CONTINUOUS, name='')
C_vars = m.addVars(tasks, lb=0.0, vtype=GRB.CONTINUOUS, name='')
Cmax_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='Cmax')
m.addConstrs((gp.quicksum((x_vars[t, p] for p in cpus)) == 1 for t in tasks), name='')
for p in cpus:
    for t in tasks:
        for t2 in tasks:
            if t == t2:
                continue
            m.addConstr(S_vars[t2] >= S_vars[t] + proc[t, p] - M * (1 - x_vars[t, p] - x_vars[t2, p] + y_vars[t, t2, p]), name='')
            m.addConstr(S_vars[t] >= S_vars[t2] + proc[t2, p] - M * (1 - x_vars[t, p] - x_vars[t2, p] + 1 - y_vars[t, t2, p]), name='')
for t in tasks:
    m.addConstr(C_vars[t] == S_vars[t] + gp.quicksum((x_vars[t, p] * proc[t, p] for p in cpus)), name='')
m.addConstrs((Cmax_var >= C_vars[t] for t in tasks), name='')
m.setObjective(Cmax_var, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
if set(tasks) != set(b_t.keys()):
    raise ValueError('Mismatch between tasks and b_t keys')
for t in tasks:
    for p in cpus:
        if (t, p) not in proc:
            raise ValueError(f'Missing proc time for task {t}, cpu {p}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')