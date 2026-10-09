LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C22", "course_name": "Operations Research: Linear Programming", "discipline": "Operations Research", "credits": "5", "interest_points": "95"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C23", "course_name": "Integer Programming", "discipline": "Operations Research", "credits": "5", "interest_points": "92"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C24", "course_name": "Stochastic Processes", "discipline": "Operations Research", "credits": "4", "interest_points": "86"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C25", "course_name": "Simulation Modeling", "discipline": "Operations Research", "credits": "4", "interest_points": "82"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C26", "course_name": "Network Flows", "discipline": "Operations Research", "credits": "4", "interest_points": "85"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C27", "course_name": "Queueing Theory", "discipline": "Operations Research", "credits": "4", "interest_points": "80"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv", "values": {"course_id": "C28", "course_name": "Revenue Management", "discipline": "Operations Research", "credits": "4", "interest_points": "88"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C22', 'course_name': 'Operations Research: Linear Programming', 'discipline': 'Operations Research', 'credits': '5', 'interest_points': '95'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C23', 'course_name': 'Integer Programming', 'discipline': 'Operations Research', 'credits': '5', 'interest_points': '92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C24', 'course_name': 'Stochastic Processes', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '86'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C25', 'course_name': 'Simulation Modeling', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '82'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C26', 'course_name': 'Network Flows', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '85'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C27', 'course_name': 'Queueing Theory', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '80'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', 'values': {'course_id': 'C28', 'course_name': 'Revenue Management', 'discipline': 'Operations Research', 'credits': '4', 'interest_points': '88'}}]
import gurobipy as gp
from gurobipy import GRB
or_courses = []
credits = {}
interest_points = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if vals.get('discipline', '') == 'Operations Research':
        cid = vals['course_id']
        or_courses.append(cid)
        try:
            credits[cid] = int(vals['credits'])
            interest_points[cid] = int(vals['interest_points'])
        except Exception as e:
            raise ValueError(f'Invalid numeric data for course {cid}: {e}')
if not set(or_courses) == set(credits) == set(interest_points):
    raise ValueError('Missing data for some Operations Research courses.')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_courses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in or_courses)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in or_courses)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')