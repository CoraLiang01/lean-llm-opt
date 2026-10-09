import gurobipy as gp
from gurobipy import GRB
time_slots = ['2:00am - 2:30am', '2:30am - 3:00am', '3:00am - 3:30am', '3:30am - 4:00am', '4:00am - 4:30am', '4:30am - 5:00am', '5:00am - 5:30am', '5:30am - 6:00am', '6:00am - 6:30am', '6:30am - 7:00am', '7:00am - 7:30am', '7:30am - 8:00am', '8:00am - 8:30am', '8:30am - 9:00am', '9:00am - 9:30am', '9:30am - 10:00am', '10:00am - 10:30am', '10:30am - 11:00am', '11:00am - 11:30am', '11:30am - 12:00pm', '12:00pm - 12:30pm', '12:30pm - 1:00pm', '1:00pm - 1:30pm', '1:30pm - 2:00pm', '2:00pm - 2:30pm', '2:30pm - 3:00pm', '3:00pm - 3:30pm', '3:30pm - 4:00pm', '4:00pm - 4:30pm', '4:30pm - 5:00pm', '5:00pm - 5:30pm', '5:30pm - 6:00pm', '6:00pm - 6:30pm', '6:30pm - 7:00pm', '7:00pm - 7:30pm', '7:30pm - 8:00pm', '8:00pm - 8:30pm', '8:30pm - 9:00pm', '9:00pm - 9:30pm', '9:30pm - 10:00pm', '10:00pm - 10:30pm', '10:30pm - 11:00pm', '11:00pm - 11:30pm', '11:30pm - 12:00am', '12:00am - 12:30am', '12:30am - 1:00am', '1:00am - 1:30am', '1:30am - 2:00am']
requirements = [2, 3, 4, 6, 5, 4, 5, 6, 7, 8, 9, 9, 8, 8, 9, 9, 10, 12, 11, 11, 12, 11, 10, 9, 8, 7, 6, 5, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 4, 3, 3, 3, 3, 4, 4, 4]
if len(time_slots) != 48 or len(requirements) != 48:
    raise ValueError('Data must have 48 time slots and 48 requirements.')
T = 48
shift_length = 16
m = gp.Model('Waitstaff_Shift_Scheduling')
x_vars = m.addVars(range(1, T + 1), vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in range(1, T + 1))), GRB.MINIMIZE)
for t in range(1, T + 1):
    m.addConstr(gp.quicksum((x_vars[(t - k - 1) % T + 1] for k in range(shift_length))) >= requirements[t - 1], name=f'demand_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for s in range(1, T + 1):
        print(f'x[{s}]: {x_vars[s].X}')
else:
    print(f'Solver status: {m.Status}')