import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',')
or_courses_df = courses_df[courses_df['discipline'].str.strip().str.casefold() == 'operations research']
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = or_courses_df['course_id'].astype(str).tolist()
credits = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['credits'].astype(int)))
interest_points = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['interest_points'].astype(int)))
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()