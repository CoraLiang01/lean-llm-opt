import gurobipy as gp
from gurobipy import GRB
time_periods = [{'Shift': 1, 'Time': '0:00-1:00', 'Number Required': 20}, {'Shift': 2, 'Time': '1:00-2:00', 'Number Required': 18}, {'Shift': 3, 'Time': '2:00-3:00', 'Number Required': 15}, {'Shift': 4, 'Time': '3:00-4:00', 'Number Required': 15}, {'Shift': 5, 'Time': '4:00-5:00', 'Number Required': 20}, {'Shift': 6, 'Time': '5:00-6:00', 'Number Required': 30}, {'Shift': 7, 'Time': '6:00-7:00', 'Number Required': 60}, {'Shift': 8, 'Time': '7:00-8:00', 'Number Required': 70}, {'Shift': 9, 'Time': '8:00-9:00', 'Number Required': 50}, {'Shift': 10, 'Time': '9:00-10:00', 'Number Required': 55}, {'Shift': 11, 'Time': '10:00-11:00', 'Number Required': 65}, {'Shift': 12, 'Time': '11:00-12:00', 'Number Required': 75}, {'Shift': 13, 'Time': '12:00-13:00', 'Number Required': 80}, {'Shift': 14, 'Time': '13:00-14:00', 'Number Required': 70}, {'Shift': 15, 'Time': '14:00-15:00', 'Number Required': 60}, {'Shift': 16, 'Time': '15:00-16:00', 'Number Required': 55}, {'Shift': 17, 'Time': '16:00-17:00', 'Number Required': 60}, {'Shift': 18, 'Time': '17:00-18:00', 'Number Required': 75}, {'Shift': 19, 'Time': '18:00-19:00', 'Number Required': 85}, {'Shift': 20, 'Time': '19:00-20:00', 'Number Required': 70}, {'Shift': 21, 'Time': '20:00-21:00', 'Number Required': 50}, {'Shift': 22, 'Time': '21:00-22:00', 'Number Required': 40}, {'Shift': 23, 'Time': '22:00-23:00', 'Number Required': 35}, {'Shift': 24, 'Time': '23:00-0:00', 'Number Required': 25}]
shifts = [tp['Shift'] for tp in time_periods]
required = {tp['Shift']: tp['Number Required'] for tp in time_periods}
m = gp.Model('bus_crew_scheduling')
x = m.addVars(shifts, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] for i in shifts)), GRB.MINIMIZE)
for t in shifts:
    indices = [(t - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[j] for j in indices)) >= required[t], name=f'c{t}')
if set(shifts) != set(required.keys()):
    raise ValueError('Mismatch between shift indices and required data.')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')