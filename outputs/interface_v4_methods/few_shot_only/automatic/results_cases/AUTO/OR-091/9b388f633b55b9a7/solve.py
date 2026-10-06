import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].astype(str).str.casefold().str.strip() == 'operations research']
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = list(or_courses_df['course_id'].astype(str))
credits = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['credits'].astype(int)))
interest_points = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['interest_points'].astype(int)))
for cid in or_course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for Operations Research course {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()