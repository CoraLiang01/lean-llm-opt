import pandas as pd
import gurobipy as gp
from gurobipy import GRB
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].str.casefold().str.strip() == 'operations research']
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = or_courses_df['course_id'].astype(str).tolist()
credits = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['credits']))
interest_points = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['interest_points']))
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in parameter keys for Operations Research courses.')
m = gp.Model('or_course_selection')
x = m.addVars(or_course_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()