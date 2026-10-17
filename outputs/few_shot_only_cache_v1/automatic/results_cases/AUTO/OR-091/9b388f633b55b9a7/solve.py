import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError('No Operations Research courses found in the data.')
course_id_col = 'course_id'
interest_points_col = 'interest_points'
credits_col = 'credits'
or_course_ids = list(or_courses_df[course_id_col])
try:
    credits = {row[course_id_col]: int(row[credits_col]) for _, row in or_courses_df.iterrows()}
    interest_points = {row[course_id_col]: int(row[interest_points_col]) for _, row in or_courses_df.iterrows()}
except Exception as e:
    raise ValueError(f'Failed to convert credits or interest_points to integer: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [i for i in or_course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected_courses))
    total_interest = sum((interest_points[i] for i in selected_courses))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for i in selected_courses:
        course_name = or_courses_df.loc[or_courses_df[course_id_col] == i, 'course_name'].values[0]
        print(f'  {i}: {course_name} (Credits: {credits[i]}, Interest Points: {interest_points[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')