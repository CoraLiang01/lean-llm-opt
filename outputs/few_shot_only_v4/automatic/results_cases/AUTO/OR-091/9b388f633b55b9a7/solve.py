import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
if 'discipline' not in courses_df.columns:
    raise KeyError("Missing required column 'discipline' in courses_42.csv")
or_mask = courses_df['discipline'].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline 'Operations Research' in courses_42.csv")
if 'course_id' in or_courses_df.columns:
    course_id_col = 'course_id'
elif 'id' in or_courses_df.columns:
    course_id_col = 'id'
elif 'CourseID' in or_courses_df.columns:
    course_id_col = 'CourseID'
else:
    course_id_col = None
if course_id_col is not None:
    or_courses_df = or_courses_df.set_index(course_id_col, drop=False)
    or_course_keys = list(or_courses_df.index)
else:
    or_course_keys = list(or_courses_df.index)

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
if 'credits' not in or_courses_df.columns:
    raise KeyError("Missing required column 'credits' in courses_42.csv")
if 'interest_points' not in or_courses_df.columns:
    raise KeyError("Missing required column 'interest_points' in courses_42.csv")
credits = safe_float(or_courses_df['credits'], 'credits')
interest_points = safe_float(or_courses_df['interest_points'], 'interest_points')
credits_dict = dict(zip(or_course_keys, credits))
interest_points_dict = dict(zip(or_course_keys, interest_points))
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points_dict[i] * x_vars[i] for i in or_course_keys)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits_dict[i] * x_vars[i] for i in or_course_keys)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = m.objVal
    selected_courses = [i for i in or_course_keys if x_vars[i].X > 0.5]
    total_credits = sum((credits_dict[i] for i in selected_courses))
    print(f'Optimal total interest points: {total_interest:.2f}')
    print(f'Total credits used: {total_credits:.2f} / 20')
    print('--- Selected Operations Research Courses ---')
    for i in selected_courses:
        row = or_courses_df.loc[i]
        course_name = row['course_name'] if 'course_name' in row else i
        print(f'  {course_name} (ID: {i}) - Credits: {credits_dict[i]}, Interest Points: {interest_points_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')