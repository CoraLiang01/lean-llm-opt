import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
discipline_target = 'operations research'
or_mask = df[discipline_col].str.strip().str.casefold() == discipline_target
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found for discipline 'Operations Research'.")
course_ids = or_courses_df['course_id'].tolist()
try:
    interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
    credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Error converting interest_points or credits to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()