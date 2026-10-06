import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',')
or_courses = df[df['discipline'].str.strip().str.casefold() == 'operations research']
if or_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = or_courses['course_id'].astype(str).tolist()
credits = dict(zip(or_courses['course_id'].astype(str), or_courses['credits']))
interest_points = dict(zip(or_courses['course_id'].astype(str), or_courses['interest_points']))
if set(course_ids) != set(credits.keys()) or set(course_ids) != set(interest_points.keys()):
    raise ValueError('Mismatch in course_ids and parameter keys for credits or interest_points.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
m.optimize()