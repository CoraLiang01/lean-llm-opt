import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = df[or_mask].copy()
course_id_col = 'course_id'
credits_col = 'credits'
interest_col = 'interest_points'
for col in [course_id_col, credits_col, interest_col]:
    if col not in or_courses_df.columns:
        raise KeyError(f"Required column '{col}' not found in Operations Research course data.")
or_course_ids = or_courses_df[course_id_col].tolist()
try:
    credits = {row[course_id_col]: int(row[credits_col]) for _, row in or_courses_df.iterrows()}
    interest_points = {row[course_id_col]: int(row[interest_col]) for _, row in or_courses_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in or_course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total value/cost: {m.objVal} (Total interest points)')
    print('--- Selected Operations Research Courses ---')
    for i in selected:
        course_name = or_courses_df.loc[or_courses_df[course_id_col] == i, 'course_name'].values[0]
        print(f'{i}: {course_name} | Credits: {credits[i]} | Interest Points: {interest_points[i]}')
    print(f'Total Credits: {total_credits}')
    print(f'Total Interest Points: {total_interest}')
else:
    print(f'No optimal solution found. Status: {m.status}')