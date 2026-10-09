import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
discipline_norm = df[discipline_col].str.strip().str.casefold()
ops_research_mask = discipline_norm == 'operations research'
ops_research_df = df[ops_research_mask].copy()
course_ids = ops_research_df['course_id'].tolist()
if len(course_ids) == 0:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
credits_dict = {}
interest_points_dict = {}
for (idx, row) in ops_research_df.iterrows():
    course_id = row['course_id']
    try:
        credits = int(row['credits'])
        interest_points = int(row['interest_points'])
    except Exception as e:
        raise ValueError(f'Non-integer value in credits or interest_points for course_id {course_id}: {e}')
    credits_dict[course_id] = credits
    interest_points_dict[course_id] = interest_points
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points_dict[c] * x_vars[c] for c in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits_dict[c] * x_vars[c] for c in course_ids)) <= 20, name='credit_limit')
m.optimize()