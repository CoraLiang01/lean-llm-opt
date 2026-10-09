import gurobipy as gp
from gurobipy import GRB
courses = [{'course_id': 'C22', 'course_name': 'Operations Research: Linear Programming', 'credits': 5, 'interest_points': 95}, {'course_id': 'C23', 'course_name': 'Integer Programming', 'credits': 5, 'interest_points': 92}, {'course_id': 'C24', 'course_name': 'Stochastic Processes', 'credits': 4, 'interest_points': 86}, {'course_id': 'C25', 'course_name': 'Simulation Modeling', 'credits': 4, 'interest_points': 82}, {'course_id': 'C26', 'course_name': 'Network Flows', 'credits': 4, 'interest_points': 85}, {'course_id': 'C27', 'course_name': 'Queueing Theory', 'credits': 4, 'interest_points': 80}, {'course_id': 'C28', 'course_name': 'Revenue Management', 'credits': 4, 'interest_points': 88}]
course_ids = [c['course_id'] for c in courses]
credits = {c['course_id']: c['credits'] for c in courses}
interest_points = {c['course_id']: c['interest_points'] for c in courses}
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')