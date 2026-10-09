import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_id_col = 'course_id'
interest_points_col = 'interest_points'
credits_col = 'credits'
or_courses_df.set_index(course_id_col, inplace=True, drop=False)

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
interest_points = safe_float(or_courses_df[interest_points_col], interest_points_col)
credits = safe_float(or_courses_df[credits_col], credits_col)
I = list(or_courses_df.index)
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in I)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in I)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in I if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total interest points: {m.objVal:.2f}')
    print(f'Total credits used: {total_credits:.2f} / 20')
    print('\nSelected Operations Research courses:')
    for i in selected:
        cname = or_courses_df.loc[i, 'course_name']
        print(f'  {i}: {cname} | Credits: {credits[i]:.1f} | Interest Points: {interest_points[i]:.1f}')
else:
    print(f'No optimal solution found. Status: {m.status}')