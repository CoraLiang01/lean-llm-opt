import gurobipy as gp
from gurobipy import GRB
courses = ['C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28']
credits = {'C22': 5, 'C23': 5, 'C24': 4, 'C25': 4, 'C26': 4, 'C27': 4, 'C28': 4}
interest_points = {'C22': 95, 'C23': 92, 'C24': 86, 'C25': 82, 'C26': 85, 'C27': 80, 'C28': 88}
if set(credits.keys()) != set(courses):
    raise ValueError('Credits data missing for some courses.')
if set(interest_points.keys()) != set(courses):
    raise ValueError('Interest points data missing for some courses.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(courses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in courses)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in courses)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')