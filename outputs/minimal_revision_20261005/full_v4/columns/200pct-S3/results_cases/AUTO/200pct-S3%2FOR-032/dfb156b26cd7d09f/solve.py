import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_filter = df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses = df[discipline_filter].copy()
if or_courses.empty:
    raise ValueError("No courses found with discipline 'Operations Research'.")
course_ids = list(or_courses['course_id'])
interest_points = dict(zip(or_courses['course_id'], or_courses['interest_points']))
credits = dict(zip(or_courses['course_id'], or_courses['credits']))
if set(interest_points.keys()) != set(course_ids) or set(credits.keys()) != set(course_ids):
    raise ValueError('Mismatch in course_ids and parameter keys for interest_points or credits.')
m = gp.Model('OR_Course_Selection')
m.setParam('MIPGap', 0.0001)
x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in course_ids:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')