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
interest_points_col = 'interest_points'
credits_col = 'credits'
or_courses_df.set_index(course_id_col, inplace=True, drop=False)
C = list(or_courses_df.index)
try:
    interest_points = or_courses_df[interest_points_col].astype(int).to_dict()
    credits = or_courses_df[credits_col].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(C, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[c] * x_vars[c] for c in C)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[c] * x_vars[c] for c in C)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [c for c in C if x_vars[c].X > 0.5]
    total_interest = sum((interest_points[c] for c in selected_courses))
    total_credits = sum((credits[c] for c in selected_courses))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('\nSelected Operations Research courses:')
    for c in selected_courses:
        cname = or_courses_df.loc[c, 'course_name']
        print(f'  {c}: {cname} (Credits: {credits[c]}, Interest Points: {interest_points[c]})')
else:
    print(f'No optimal solution found. Status: {m.status}')