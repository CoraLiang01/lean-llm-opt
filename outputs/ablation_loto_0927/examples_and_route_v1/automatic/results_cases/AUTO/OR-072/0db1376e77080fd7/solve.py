LEGACY_OBSERVATION = 'Shift,Time,Number Required\n1,0:00-1:00,20\n2,1:00-2:00,18\n3,2:00-3:00,15\n4,3:00-4:00,15\n5,4:00-5:00,20\n6,5:00-6:00,30\n7,6:00-7:00,60\n8,7:00-8:00,70\n9,8:00-9:00,50\n10,9:00-10:00,55\n11,10:00-11:00,65\n12,11:00-12:00,75\n13,12:00-13:00,80\n14,13:00-14:00,70\n15,14:00-15:00,60\n16,15:00-16:00,55\n17,16:00-17:00,60\n18,17:00-18:00,75\n19,18:00-19:00,85\n20,19:00-20:00,70\n21,20:00-21:00,50\n22,21:00-22:00,40\n23,22:00-23:00,35\n24,23:00-0:00,25'
LEGACY_RECORDS = [{'source': '', 'values': {'Shift': '1', 'Time': '0:00-1:00', 'Number Required': '20'}}, {'source': '', 'values': {'Shift': '2', 'Time': '1:00-2:00', 'Number Required': '18'}}, {'source': '', 'values': {'Shift': '3', 'Time': '2:00-3:00', 'Number Required': '15'}}, {'source': '', 'values': {'Shift': '4', 'Time': '3:00-4:00', 'Number Required': '15'}}, {'source': '', 'values': {'Shift': '5', 'Time': '4:00-5:00', 'Number Required': '20'}}, {'source': '', 'values': {'Shift': '6', 'Time': '5:00-6:00', 'Number Required': '30'}}, {'source': '', 'values': {'Shift': '7', 'Time': '6:00-7:00', 'Number Required': '60'}}, {'source': '', 'values': {'Shift': '8', 'Time': '7:00-8:00', 'Number Required': '70'}}, {'source': '', 'values': {'Shift': '9', 'Time': '8:00-9:00', 'Number Required': '50'}}, {'source': '', 'values': {'Shift': '10', 'Time': '9:00-10:00', 'Number Required': '55'}}, {'source': '', 'values': {'Shift': '11', 'Time': '10:00-11:00', 'Number Required': '65'}}, {'source': '', 'values': {'Shift': '12', 'Time': '11:00-12:00', 'Number Required': '75'}}, {'source': '', 'values': {'Shift': '13', 'Time': '12:00-13:00', 'Number Required': '80'}}, {'source': '', 'values': {'Shift': '14', 'Time': '13:00-14:00', 'Number Required': '70'}}, {'source': '', 'values': {'Shift': '15', 'Time': '14:00-15:00', 'Number Required': '60'}}, {'source': '', 'values': {'Shift': '16', 'Time': '15:00-16:00', 'Number Required': '55'}}, {'source': '', 'values': {'Shift': '17', 'Time': '16:00-17:00', 'Number Required': '60'}}, {'source': '', 'values': {'Shift': '18', 'Time': '17:00-18:00', 'Number Required': '75'}}, {'source': '', 'values': {'Shift': '19', 'Time': '18:00-19:00', 'Number Required': '85'}}, {'source': '', 'values': {'Shift': '20', 'Time': '19:00-20:00', 'Number Required': '70'}}, {'source': '', 'values': {'Shift': '21', 'Time': '20:00-21:00', 'Number Required': '50'}}, {'source': '', 'values': {'Shift': '22', 'Time': '21:00-22:00', 'Number Required': '40'}}, {'source': '', 'values': {'Shift': '23', 'Time': '22:00-23:00', 'Number Required': '35'}}, {'source': '', 'values': {'Shift': '24', 'Time': '23:00-0:00', 'Number Required': '25'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
shifts = []
required = {}
for rec in records:
    vals = rec['values']
    shift = int(vals['Shift'])
    req = int(vals['Number Required'])
    shifts.append(shift)
    required[shift] = req
if sorted(shifts) != list(range(1, 25)):
    raise ValueError('Missing or duplicate shift indices in LEGACY_RECORDS.')
m = gp.Model('BusCrewScheduling')
x = m.addVars(shifts, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((x[t] for t in shifts)), GRB.MINIMIZE)
for s in shifts:
    indices = [(s - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in indices)) >= required[s], name=f'cover_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')