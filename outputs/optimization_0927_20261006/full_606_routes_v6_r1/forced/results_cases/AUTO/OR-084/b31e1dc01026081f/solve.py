LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv", "values": {"Process": "BI", "1": "1.1", "2": "2.1", "3": "3", "4": "1", "5": "0.7", "6": "5", "7": "3", "8": "3.5", "9": "4.4", "10": "3.8", "11": "3.5", "12": "2.8", "13": "4.1", "14": "2.9", "15": "5.4", "16": "5.8", "17": "2.6", "18": "4.9", "19": "3.4", "20": "3.6", "21": "5.6", "22": "0.9", "23": "1", "24": "0.6", "25": "5.1", "26": "4.8", "27": "5.3", "28": "5.9", "29": "4.9", "30": "3", "31": "4.8", "32": "1.2", "33": "4", "34": "1.3", "35": "5.7", "36": "3.4", "37": "2.8", "38": "2", "39": "4.8", "40": "3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
record = next((r for r in LEGACY_RECORDS if r['source'].endswith('18.csv')))
bt_dict = {int(k): float(v) for (k, v) in record['values'].items() if k.isdigit()}
tasks = sorted(bt_dict.keys())
processors = [1, 2, 3]
frequencies = {1: 1.33, 2: 2.0, 3: 2.66}
if set(tasks) != set(range(1, 41)):
    raise ValueError('Task indices in data do not match required set 1..40')
if set(processors) != set(frequencies.keys()):
    raise ValueError('Processor indices in frequencies do not match required set 1..3')
proc_time = {(t, p): bt_dict[t] / frequencies[p] for t in tasks for p in processors}
M = sum((bt_dict[t] / min(frequencies.values()) for t in tasks)) + 1
m = gp.Model('TaskAssignment_Makespan')
x_vars = m.addVars(tasks, processors, vtype=GRB.BINARY, name='')
s_vars = m.addVars(tasks, lb=0.0, vtype=GRB.CONTINUOUS, name='')
C_vars = m.addVars(tasks, lb=0.0, vtype=GRB.CONTINUOUS, name='')
Cmax_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='Cmax')
y_vars = m.addVars(((t, tp, p) for t in tasks for tp in tasks if t != tp for p in processors), vtype=GRB.BINARY, name='')
m.addConstrs((gp.quicksum((x_vars[t, p] for p in processors)) == 1 for t in tasks), name='')
m.addConstrs((C_vars[t] == s_vars[t] + gp.quicksum((proc_time[t, p] * x_vars[t, p] for p in processors)) for t in tasks), name='')
for p in processors:
    for t in tasks:
        for tp in tasks:
            if t == tp:
                continue
            m.addConstr(s_vars[tp] >= C_vars[t] - M * (1 - y_vars[t, tp, p]) - M * (1 - x_vars[t, p]) - M * (1 - x_vars[tp, p]), name='')
            m.addConstr(s_vars[t] >= C_vars[tp] - M * y_vars[t, tp, p] - M * (1 - x_vars[t, p]) - M * (1 - x_vars[tp, p]), name='')
            m.addConstr(y_vars[t, tp, p] + y_vars[tp, t, p] == 1, name='')
m.addConstrs((Cmax_var >= C_vars[t] for t in tasks), name='')
m.setObjective(Cmax_var, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')