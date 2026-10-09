import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_id_col = 'course_id'
credits_col = 'credits'
interest_col = 'interest_points'
or_course_ids = or_courses_df[course_id_col].tolist()
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row[course_id_col]
    try:
        credits[cid] = int(row[credits_col])
        interest_points[cid] = int(row[interest_col])
    except Exception as e:
        raise ValueError(f'Invalid numeric value for course_id {cid}: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[i] for i in or_course_ids if x_vars[i].X > 0.5))
    total_credits = sum((credits[i] for i in or_course_ids if x_vars[i].X > 0.5))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('--- Selected Operations Research Courses ---')
    for i in or_course_ids:
        if x_vars[i].X > 0.5:
            row = or_courses_df[or_courses_df[course_id_col] == i].iloc[0]
            print(f"{row[course_id_col]}: {row['course_name']} | Credits: {credits[i]} | Interest Points: {interest_points[i]}")
else:
    print(f'No optimal solution found. Status: {m.status}')