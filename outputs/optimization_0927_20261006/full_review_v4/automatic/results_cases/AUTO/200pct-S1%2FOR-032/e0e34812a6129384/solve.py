import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_courses_df['course_id'] = or_courses_df['course_id'].astype(str)
I = list(or_courses_df['course_id'])
try:
    credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
    interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'credits' or 'interest_points' to int: {e}")
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in I)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in I)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in I if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits} / 20')
    print('Selected Operations Research courses:')
    for i in selected:
        cname = or_courses_df.loc[or_courses_df['course_id'] == i, 'course_name'].values[0]
        print(f'  {i}: {cname} (Credits: {credits[i]}, Interest Points: {interest_points[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')