LEGACY_OBSERVATION = 'course_id,course_name,discipline,credits,interest_points\nC22,Operations Research: Linear Programming,Operations Research,5,95\nC23,Integer Programming,Operations Research,5,92\nC24,Stochastic Processes,Operations Research,4,86\nC25,Simulation Modeling,Operations Research,4,82\nC26,Network Flows,Operations Research,4,85\nC27,Queueing Theory,Operations Research,4,80\nC28,Revenue Management,Operations Research,4,88'
LEGACY_RECORDS = [{'source': '', 'values': {'course_id': 'C22', 'course_name': 'Operations Research: Linear Programming', 'discipline': 'Operations Research', 'credits': '5', 'interest_points': '95'}}, {'source': '', 'values': {'course_id': 'C23', 'course_name': 'Integer Programming', 'discipline': 'Operations Research', 'credits': '5', 'interest_points': '92'}}, {'source': '', 'values': {'course_id': 'C24', 'course_name': 'Stochastic Processes', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '86'}}, {'source': '', 'values': {'course_id': 'C25', 'course_name': 'Simulation Modeling', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '82'}}, {'source': '', 'values': {'course_id': 'C26', 'course_name': 'Network Flows', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '85'}}, {'source': '', 'values': {'course_id': 'C27', 'course_name': 'Queueing Theory', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '80'}}, {'source': '', 'values': {'course_id': 'C28', 'course_name': 'Revenue Management', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '88'}}]
import gurobipy as gp
from gurobipy import GRB
or_courses = []
credits = {}
interest_points = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if vals.get('discipline') == 'Operations Research':
        cid = vals['course_id']
        or_courses.append(cid)
        credits[cid] = int(vals['credits'])
        interest_points[cid] = int(vals['interest_points'])
if set(credits) != set(or_courses) or set(interest_points) != set(or_courses):
    raise ValueError('Missing data for some Operations Research courses.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_courses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in or_courses)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in or_courses)) <= 20, name='credit_cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')