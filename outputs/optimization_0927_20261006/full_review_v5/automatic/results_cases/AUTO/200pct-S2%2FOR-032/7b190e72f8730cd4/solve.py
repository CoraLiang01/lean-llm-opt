import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_id_col = 'course_id'
credits_col = 'credits'
interest_points_col = 'interest_points'
or_courses_df.set_index(course_id_col, inplace=True, drop=False)
or_courses_df[credits_col] = or_courses_df[credits_col].astype(int)
or_courses_df[interest_points_col] = or_courses_df[interest_points_col].astype(int)
or_course_ids = list(or_courses_df.index)
credits = or_courses_df[credits_col].to_dict()
interest_points = or_courses_df[interest_points_col].to_dict()
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in or_course_ids)) <= 20, name='TotalCredits')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in or_course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('\nSelected Operations Research courses:')
    for i in selected:
        cname = or_courses_df.loc[i, 'course_name']
        print(f'  {i}: {cname} (Credits: {credits[i]}, Interest Points: {interest_points[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')