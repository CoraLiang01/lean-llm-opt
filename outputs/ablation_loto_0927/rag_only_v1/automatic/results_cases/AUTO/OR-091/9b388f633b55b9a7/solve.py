import pandas as pd
import numpy as np
from gurobipy import Model, GRB
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].str.casefold().str.strip() == 'operations research']
OR_COURSES = or_courses_df['course_id'].astype(str).tolist()
credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
for cid in OR_COURSES:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for course_id {cid}')
m = Model('or_course_selection')
x = m.addVars(OR_COURSES, vtype=GRB.BINARY, name='')
m.setObjective(sum((interest_points[cid] * x[cid] for cid in OR_COURSES)), GRB.MAXIMIZE)
m.addConstr(sum((credits[cid] * x[cid] for cid in OR_COURSES)) <= 20, name='credit_limit')
m.optimize()