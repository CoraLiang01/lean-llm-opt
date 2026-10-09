import gurobipy as gp
from gurobipy import GRB
participants = ['Carpenter', 'Electrician', 'Painter', 'Worker_004', 'Worker_005']
d_ji = [[2, 3, 1, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 1, 3, 2, 2]]
N = len(participants)
for i in range(N):
    total_days = sum((d_ji[j][i] for j in range(N)))
    if total_days != 10:
        raise ValueError(f'Worker {participants[i]} has total days {total_days}, expected 10.')
if any((len(row) != N for row in d_ji)):
    raise ValueError('d_ji matrix is not square or does not match number of participants.')
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(participants, lb=0.0, name='')
m.addConstr(wage_vars[participants[0]] == 60.0, name='fix_w1')
for k in range(N):
    lhs = gp.quicksum((d_ji[k][i] * wage_vars[participants[i]] for i in range(N)))
    rhs = 10 * wage_vars[participants[k]]
    m.addConstr(lhs == rhs, name=f'balance_{participants[k]}')
m.setObjective(0, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in wage_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')