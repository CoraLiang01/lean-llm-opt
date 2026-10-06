import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].astype(str).str.casefold().str.strip() == 'operations research']
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
OR_COURSES = list(or_courses_df['course_id'])
credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
if set(credits.keys()) != set(OR_COURSES) or set(interest_points.keys()) != set(OR_COURSES):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(OR_COURSES, vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in OR_COURSES)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in OR_COURSES)) <= 20, name='TotalCreditsLimit')
m.optimize()