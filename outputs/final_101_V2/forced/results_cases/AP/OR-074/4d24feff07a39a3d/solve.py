import gurobipy as gp
from gurobipy import GRB
intervals = ['2:00am - 2:30am', '2:30am - 3:00am', '3:00am - 3:30am', '3:30am - 4:00am', '4:00am - 4:30am', '4:30am - 5:00am', '5:00am - 5:30am', '5:30am - 6:00am', '6:00am - 6:30am', '6:30am - 7:00am', '7:00am - 7:30am', '7:30am - 8:00am', '8:00am - 8:30am', '8:30am - 9:00am', '9:00am - 9:30am', '9:30am - 10:00am', '10:00am - 10:30am', '10:30am - 11:00am', '11:00am - 11:30am', '11:30am - 12:00pm', '12:00pm - 12:30pm', '12:30pm - 1:00pm', '1:00pm - 1:30pm', '1:30pm - 2:00pm', '2:00pm - 2:30pm', '2:30pm - 3:00pm', '3:00pm - 3:30pm', '3:30pm - 4:00pm', '4:00pm - 4:30pm', '4:30pm - 5:00pm', '5:00pm - 5:30pm', '5:30pm - 6:00pm', '6:00pm - 6:30pm', '6:30pm - 7:00pm', '7:00pm - 7:30pm', '7:30pm - 8:00pm', '8:00pm - 8:30pm', '8:30pm - 9:00pm', '9:00pm - 9:30pm', '9:30pm - 10:00pm', '10:00pm - 10:30pm', '10:30pm - 11:00pm', '11:00pm - 11:30pm', '11:30pm - 12:00am', '12:00am - 12:30am', '12:30am - 1:00am', '1:00am - 1:30am', '1:30am - 2:00am']
requirement = [{'interval': '2:00am - 2:30am', 'value': 2}, {'interval': '2:30am - 3:00am', 'value': 3}, {'interval': '3:00am - 3:30am', 'value': 4}, {'interval': '3:30am - 4:00am', 'value': 6}, {'interval': '4:00am - 4:30am', 'value': 5}, {'interval': '4:30am - 5:00am', 'value': 4}, {'interval': '5:00am - 5:30am', 'value': 5}, {'interval': '5:30am - 6:00am', 'value': 6}, {'interval': '6:00am - 6:30am', 'value': 7}, {'interval': '6:30am - 7:00am', 'value': 8}, {'interval': '7:00am - 7:30am', 'value': 9}, {'interval': '7:30am - 8:00am', 'value': 9}, {'interval': '8:00am - 8:30am', 'value': 8}, {'interval': '8:30am - 9:00am', 'value': 8}, {'interval': '9:00am - 9:30am', 'value': 9}, {'interval': '9:30am - 10:00am', 'value': 9}, {'interval': '10:00am - 10:30am', 'value': 10}, {'interval': '10:30am - 11:00am', 'value': 12}, {'interval': '11:00am - 11:30am', 'value': 11}, {'interval': '11:30am - 12:00pm', 'value': 11}, {'interval': '12:00pm - 12:30pm', 'value': 12}, {'interval': '12:30pm - 1:00pm', 'value': 11}, {'interval': '1:00pm - 1:30pm', 'value': 10}, {'interval': '1:30pm - 2:00pm', 'value': 9}, {'interval': '2:00pm - 2:30pm', 'value': 8}, {'interval': '2:30pm - 3:00pm', 'value': 7}, {'interval': '3:00pm - 3:30pm', 'value': 6}, {'interval': '3:30pm - 4:00pm', 'value': 5}, {'interval': '4:00pm - 4:30pm', 'value': 5}, {'interval': '4:30pm - 5:00pm', 'value': 6}, {'interval': '5:00pm - 5:30pm', 'value': 7}, {'interval': '5:30pm - 6:00pm', 'value': 8}, {'interval': '6:00pm - 6:30pm', 'value': 9}, {'interval': '6:30pm - 7:00pm', 'value': 10}, {'interval': '7:00pm - 7:30pm', 'value': 9}, {'interval': '7:30pm - 8:00pm', 'value': 8}, {'interval': '8:00pm - 8:30pm', 'value': 7}, {'interval': '8:30pm - 9:00pm', 'value': 6}, {'interval': '9:00pm - 9:30pm', 'value': 5}, {'interval': '9:30pm - 10:00pm', 'value': 4}, {'interval': '10:00pm - 10:30pm', 'value': 4}, {'interval': '10:30pm - 11:00pm', 'value': 3}, {'interval': '11:00pm - 11:30pm', 'value': 3}, {'interval': '11:30pm - 12:00am', 'value': 3}, {'interval': '12:00am - 12:30am', 'value': 3}, {'interval': '12:30am - 1:00am', 'value': 4}, {'interval': '1:00am - 1:30am', 'value': 4}, {'interval': '1:30am - 2:00am', 'value': 4}]
if len(intervals) != 48 or len(requirement) != 48:
    raise ValueError('There must be exactly 48 intervals and 48 requirements.')
interval_indices = list(range(1, 49))
interval_name = {i: intervals[i - 1] for i in interval_indices}
r = {i: requirement[i - 1]['value'] for i in interval_indices}
m = gp.Model('Waitstaff_Scheduling')
x = m.addVars(interval_indices, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in interval_indices)), GRB.MINIMIZE)
for s in interval_indices:
    m.addConstr(gp.quicksum((x[(s - k - 1) % 48 + 1] for k in range(16))) >= r[s], name=f'cover_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for t in interval_indices:
        print(f'x[{t}]: {x[t].X}')
else:
    print(f'Solver status: {m.Status}')