LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv", "values": {"Process": "BI", "1": "1.1", "2": "2.1", "3": "3", "4": "1", "5": "0.7", "6": "5", "7": "3", "8": "3.5", "9": "4.4", "10": "3.8", "11": "3.5", "12": "2.8", "13": "4.1", "14": "2.9", "15": "5.4", "16": "5.8", "17": "2.6", "18": "4.9", "19": "3.4", "20": "3.6", "21": "5.6", "22": "0.9", "23": "1", "24": "0.6", "25": "5.1", "26": "4.8", "27": "5.3", "28": "5.9", "29": "4.9", "30": "3", "31": "4.8", "32": "1.2", "33": "4", "34": "1.3", "35": "5.7", "36": "3.4", "37": "2.8", "38": "2", "39": "4.8", "40": "3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv', 'values': {'Process': 'BI', '1': '1.1', '2': '2.1', '3': '3', '4': '1', '5': '0.7', '6': '5', '7': '3', '8': '3.5', '9': '4.4', '10': '3.8', '11': '3.5', '12': '2.8', '13': '4.1', '14': '2.9', '15': '5.4', '16': '5.8', '17': '2.6', '18': '4.9', '19': '3.4', '20': '3.6', '21': '5.6', '22': '0.9', '23': '1', '24': '0.6', '25': '5.1', '26': '4.8', '27': '5.3', '28': '5.9', '29': '4.9', '30': '3', '31': '4.8', '32': '1.2', '33': '4', '34': '1.3', '35': '5.7', '36': '3.4', '37': '2.8', '38': '2', '39': '4.8', '40': '3'}}]
import gurobipy as gp
from gurobipy import GRB
record = next((r for r in LEGACY_RECORDS if r['source']))
a_j_dict = {int(k): float(v) for k, v in record['values'].items() if k.isdigit()}
cpus = [1, 2, 3]
cpu_freq = {1: 1.33, 2: 2.0, 3: 2.66}
tasks = sorted(a_j_dict.keys())
if set(tasks) != set(range(1, 41)):
    raise ValueError('Missing or extra task indices in instruction counts.')
p_ij = {(i, j): a_j_dict[j] / cpu_freq[i] for i in cpus for j in tasks}
m = gp.Model('TaskAssignment')
x_vars = m.addVars(cpus, tasks, vtype=GRB.BINARY, name='')
Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Cmax')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in cpus)) == 1 for j in tasks), name='')
m.addConstrs((gp.quicksum((p_ij[i, j] * x_vars[i, j] for j in tasks)) <= Cmax for i in cpus), name='')
m.setObjective(Cmax, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')