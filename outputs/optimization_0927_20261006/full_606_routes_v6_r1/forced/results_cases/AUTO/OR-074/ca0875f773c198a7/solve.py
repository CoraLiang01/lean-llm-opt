import gurobipy as gp
from gurobipy import GRB
N = 48
periods = [{'Time': '2:00am - 2:30am', 'Requirement': 2}, {'Time': '2:30am - 3:00am', 'Requirement': 3}, {'Time': '3:00am - 3:30am', 'Requirement': 4}, {'Time': '3:00am - 3:30am', 'Requirement': 4}, {'Time': '3:30am - 4:00am', 'Requirement': 6}, {'Time': '4:00am - 4:30am', 'Requirement': 5}, {'Time': '4:30am - 5:00am', 'Requirement': 4}, {'Time': '5:00am - 5:30am', 'Requirement': 5}, {'Time': '5:30am - 6:00am', 'Requirement': 6}, {'Time': '6:00am - 6:30am', 'Requirement': 7}, {'Time': '6:30am - 7:00am', 'Requirement': 8}, {'Time': '7:00am - 7:30am', 'Requirement': 9}, {'Time': '7:30am - 8:00am', 'Requirement': 9}, {'Time': '8:00am - 8:30am', 'Requirement': 8}, {'Time': '8:30am - 9:00am', 'Requirement': 8}, {'Time': '9:00am - 9:30am', 'Requirement': 9}, {'Time': '9:30am - 10:00am', 'Requirement': 9}, {'Time': '10:00am - 10:30am', 'Requirement': 10}, {'Time': '10:30am - 11:00am', 'Requirement': 12}, {'Time': '11:00am - 11:30am', 'Requirement': 11}, {'Time': '11:30am - 12:00pm', 'Requirement': 11}, {'Time': '12:00pm - 12:30pm', 'Requirement': 12}, {'Time': '12:30pm - 1:00pm', 'Requirement': 11}, {'Time': '1:00pm - 1:30pm', 'Requirement': 10}, {'Time': '1:30pm - 2:00pm', 'Requirement': 9}, {'Time': '2:00pm - 2:30pm', 'Requirement': 8}, {'Time': '2:30pm - 3:00pm', 'Requirement': 7}, {'Time': '3:00pm - 3:30pm', 'Requirement': 6}, {'Time': '3:30pm - 4:00pm', 'Requirement': 5}, {'Time': '4:00pm - 4:30pm', 'Requirement': 5}, {'Time': '4:30pm - 5:00pm', 'Requirement': 6}, {'Time': '5:00pm - 5:30pm', 'Requirement': 7}, {'Time': '5:30pm - 6:00pm', 'Requirement': 8}, {'Time': '6:00pm - 6:30pm', 'Requirement': 9}, {'Time': '6:30pm - 7:00pm', 'Requirement': 10}, {'Time': '7:00pm - 7:30pm', 'Requirement': 9}, {'Time': '7:30pm - 8:00pm', 'Requirement': 8}, {'Time': '8:00pm - 8:30pm', 'Requirement': 7}, {'Time': '8:30pm - 9:00pm', 'Requirement': 6}, {'Time': '9:00pm - 9:30pm', 'Requirement': 5}, {'Time': '9:30pm - 10:00pm', 'Requirement': 4}, {'Time': '10:00pm - 10:30pm', 'Requirement': 4}, {'Time': '10:30pm - 11:00pm', 'Requirement': 3}, {'Time': '11:00pm - 11:30pm', 'Requirement': 3}, {'Time': '11:30pm - 12:00am', 'Requirement': 3}, {'Time': '12:00am - 12:30am', 'Requirement': 3}, {'Time': '12:30am - 1:00am', 'Requirement': 4}, {'Time': '1:00am - 1:30am', 'Requirement': 4}, {'Time': '1:30am - 2:00am', 'Requirement': 4}]
if len(periods) != N:
    raise ValueError(f'Expected {N} periods, got {len(periods)}')
r = {s + 1: periods[s]['Requirement'] for s in range(N)}
m = gp.Model('Waitstaff_Scheduling')
x_vars = m.addVars(range(1, N + 1), vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in range(1, N + 1))), GRB.MINIMIZE)
for s in range(1, N + 1):
    m.addConstr(gp.quicksum((x_vars[(s - k - 1) % N + 1] for k in range(16))) >= r[s], name=f'cover_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')