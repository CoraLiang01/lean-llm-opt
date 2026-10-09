LEGACY_OBSERVATION = '{"values": {"course_id": "C22", "course_name": "Operations Research: Linear Programming", "discipline": "Operations Research", "credits": "5", "interest_points": "95"}}\n{"values": {"course_id": "C23", "course_name": "Integer Programming", "discipline": "Operations Research", "credits": "5", "interest_points": "92"}}\n{"values": {"course_id": "C24", "course_name": "Stochastic Processes", "discipline": "Operations Research", "credits": "4", "interest_points": "86"}}\n{"values": {"course_id": "C25", "course_name": "Simulation Modeling", "discipline": "Operations Research", "credits": "4", "interest_points": "82"}}\n{"values": {"course_id": "C26", "course_name": "Network Flows", "discipline": "Operations Research", "credits": "4", "interest_points": "85"}}\n{"values": {"course_id": "C27", "course_name": "Queueing Theory", "discipline": "Operations Research", "credits": "4", "interest_points": "80"}}\n{"values": {"course_id": "C28", "course_name": "Revenue Management", "discipline": "Operations Research", "credits": "4", "interest_points": "88"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'course_id': 'C22', 'course_name': 'Operations Research: Linear Programming', 'discipline': 'Operations Research', 'credits': '5', 'interest_points': '95'}}, {'source': '', 'values': {'course_id': 'C23', 'course_name': 'Integer Programming', 'discipline': 'Operations Research', 'credits': '5', 'interest_points': '92'}}, {'source': '', 'values': {'course_id': 'C24', 'course_name': 'Stochastic Processes', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '86'}}, {'source': '', 'values': {'course_id': 'C25', 'course_name': 'Simulation Modeling', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '82'}}, {'source': '', 'values': {'course_id': 'C26', 'course_name': 'Network Flows', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '85'}}, {'source': '', 'values': {'course_id': 'C27', 'course_name': 'Queueing Theory', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '80'}}, {'source': '', 'values': {'course_id': 'C28', 'course_name': 'Revenue Management', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '88'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
or_courses = [r['values'] for r in records if r['values'].get('discipline') == 'Operations Research']
course_ids = [c['course_id'] for c in or_courses]
credits = {c['course_id']: int(c['credits']) for c in or_courses}
interest_points = {c['course_id']: int(c['interest_points']) for c in or_courses}
if set(credits.keys()) != set(course_ids) or set(interest_points.keys()) != set(course_ids):
    raise ValueError('Missing coefficients for some course_ids.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(course_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')