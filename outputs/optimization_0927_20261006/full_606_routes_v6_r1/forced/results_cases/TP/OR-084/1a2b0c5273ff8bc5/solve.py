import gurobipy as gp
from gurobipy import GRB
tasks = [f'T{t}' for t in range(1, 41)]
cpus = ['CPU1', 'CPU2', 'CPU3']
frequencies = {'CPU1': 1.33, 'CPU2': 2.0, 'CPU3': 2.66}
b_t = {'T1': 1.1, 'T2': 2.1, 'T3': 3, 'T4': 1, 'T5': 0.7, 'T6': 5, 'T7': 3, 'T8': 3.5, 'T9': 4.4, 'T10': 3.8, 'T11': 3.5, 'T12': 2.8, 'T13': 4.1, 'T14': 2.9, 'T15': 5.4, 'T16': 5.8, 'T17': 2.6, 'T18': 4.9, 'T19': 3.4, 'T20': 3.6, 'T21': 5.6, 'T22': 0.9, 'T23': 1, 'T24': 0.6, 'T25': 5.1, 'T26': 4.8, 'T27': 5.3, 'T28': 5.9, 'T29': 4.9, 'T30': 3, 'T31': 4.8, 'T32': 1.2, 'T33': 4, 'T34': 1.3, 'T35': 5.7, 'T36': 3.4, 'T37': 2.8, 'T38': 2, 'T39': 4.8, 'T40': 3}
if set(b_t.keys()) != set(tasks):
    raise ValueError('b_t keys do not match tasks set')
if set(frequencies.keys()) != set(cpus):
    raise ValueError('frequencies keys do not match cpus set')
m = gp.Model('TaskAssignment')
x_vars = m.addVars(tasks, cpus, vtype=GRB.BINARY, name='')
Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Cmax')
m.setObjective(Cmax, GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[t, p] for p in cpus)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((b_t[t] / frequencies[p] * x_vars[t, p] for t in tasks)) <= Cmax for p in cpus), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')