import gurobipy as gp
from gurobipy import GRB
workstations = [1, 2, 3]
models = list(range(1, 102))
a_wm = {1: [6, 4, 6, 7, 5, 8, 6, 5, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 6, 5, 7, 6, 8, 7, 5, 6, 7, 8, 9], 2: [5, 5, 5, 6, 4, 7, 5, 4, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 5, 4, 6, 5, 7, 6, 4, 5, 6, 7, 3], 3: [4, 6, 5, 5, 6, 7, 4, 6, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 4, 6, 5, 4, 7, 5, 6, 4, 5, 7, 6]}
for w in workstations:
    if len(a_wm[w]) != 101:
        raise ValueError(f'Workstation {w} has {len(a_wm[w])} processing times, expected 101.')
C_w = {1: 1296, 2: 1238.4, 3: 1267.2}
m = gp.Model('Radio_Idle_Min')
x_vars = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
I_vars = m.addVars(workstations, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((I_vars[w] for w in workstations)), GRB.MINIMIZE)
for w in workstations:
    m.addConstr(gp.quicksum((a_wm[w][m_idx - 1] * x_vars[m_idx] for m_idx in models)) + I_vars[w] == C_w[w], name=f'cap_w{w}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')