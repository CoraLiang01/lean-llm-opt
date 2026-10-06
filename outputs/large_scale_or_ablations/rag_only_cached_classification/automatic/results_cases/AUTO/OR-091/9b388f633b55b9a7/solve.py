import pandas as pd
import numpy as np
from gurobipy import Model, GRB
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
discipline_norm = courses_df['discipline'].str.casefold().str.strip()
or_mask = discipline_norm == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
OR_COURSES = list(or_courses_df['course_id'])
credits = dict(zip(or_courses_df['course_id'], or_courses_df['credits']))
interest_points = dict(zip(or_courses_df['course_id'], or_courses_df['interest_points']))
for cid in OR_COURSES:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for course_id {cid} in credits or interest_points.')
m = Model('or_course_selection')
x = m.addVars(OR_COURSES, vtype=GRB.BINARY, name='')
m.setObjective(sum((interest_points[i] * x[i] for i in OR_COURSES)), GRB.MAXIMIZE)
m.addConstr(sum((credits[i] * x[i] for i in OR_COURSES)) <= 20, name='credit_limit')
m.optimize()