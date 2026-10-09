import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
discipline_col = courses_df['discipline'].str.casefold().str.strip()
or_mask = discipline_col == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = or_courses_df['course_id'].tolist()
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = str(row['course_id'])
    try:
        credits[cid] = int(row['credits'])
        interest_points[cid] = int(row['interest_points'])
    except Exception as e:
        raise ValueError(f'Invalid numeric data for course_id {cid}: {e}')
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
m = Model('or_course_selection')
x_vars = m.addVars(or_course_ids, vtype=GRB.BINARY, name='')
m.setObjective(quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), GRB.MAXIMIZE)
m.addConstr(quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()