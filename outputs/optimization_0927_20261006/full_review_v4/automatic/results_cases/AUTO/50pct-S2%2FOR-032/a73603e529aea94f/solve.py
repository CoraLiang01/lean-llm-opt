import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_courses_df['course_id'] = or_courses_df['course_id'].astype(str)
I = list(or_courses_df['course_id'])
try:
    credits_dict = dict(zip(or_courses_df['course_id'], or_courses_df['credits'].astype(int)))
    interest_points_dict = dict(zip(or_courses_df['course_id'], or_courses_df['interest_points'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points_dict[i] * x_vars[i] for i in I)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits_dict[i] * x_vars[i] for i in I)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in I if x_vars[i].X > 0.5]
    total_credits = sum((credits_dict[i] for i in selected))
    total_interest = sum((interest_points_dict[i] for i in selected))
    print(f'Optimal total value/cost: {m.objVal} (Total interest points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for i in selected:
        row = or_courses_df.loc[or_courses_df['course_id'] == i].iloc[0]
        print(f"  {row['course_id']}: {row['course_name']} | Credits: {credits_dict[i]}, Interest Points: {interest_points_dict[i]}")
    print(f'\nTotal selected: {len(selected)} courses')
else:
    print(f'No optimal solution found. Status: {m.status}')