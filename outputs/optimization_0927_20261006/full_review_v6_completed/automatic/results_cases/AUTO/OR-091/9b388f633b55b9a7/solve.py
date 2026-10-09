import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'.casefold()
ops_research_df = df[ops_research_mask].copy()
if ops_research_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
ops_course_ids = ops_research_df['course_id'].tolist()
credits_dict = dict(zip(ops_research_df['course_id'], ops_research_df['credits'].astype(int)))
interest_points_dict = dict(zip(ops_research_df['course_id'], ops_research_df['interest_points'].astype(int)))
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(ops_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points_dict[i] * x_vars[i] for i in ops_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits_dict[i] * x_vars[i] for i in ops_course_ids)) <= 20, name='TotalCredits')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [i for i in ops_course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits_dict[i] for i in selected_courses))
    total_interest = sum((interest_points_dict[i] for i in selected_courses))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits} / 20')
    print('--- Selected Operations Research Courses ---')
    for i in selected_courses:
        course_row = ops_research_df[ops_research_df['course_id'] == i].iloc[0]
        print(f"{i}: {course_row['course_name']} | Credits: {credits_dict[i]} | Interest Points: {interest_points_dict[i]}")
else:
    print(f'No optimal solution found. Status: {m.status}')