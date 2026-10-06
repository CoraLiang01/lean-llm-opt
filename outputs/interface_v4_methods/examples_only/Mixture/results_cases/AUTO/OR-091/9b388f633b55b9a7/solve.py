import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].str.casefold().str.strip() == 'operations research']
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = list(or_courses_df['course_id'])
credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
for cid in or_course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for course_id {cid}.')
CREDIT_LIMIT = 20
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= CREDIT_LIMIT, name='credit_limit')
m.optimize()