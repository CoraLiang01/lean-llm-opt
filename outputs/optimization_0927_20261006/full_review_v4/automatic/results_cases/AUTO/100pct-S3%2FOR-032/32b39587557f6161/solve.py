import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_df = df[ops_research_mask].copy()
if ops_research_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_df['course_id'].tolist()
try:
    interest_points = ops_research_df.set_index('course_id')['interest_points'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'interest_points' to int: {e}")
try:
    credits = ops_research_df.set_index('course_id')['credits'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'credits' to int: {e}")
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[i] for i in course_ids if x_vars[i].X > 0.5))
    total_credits = sum((credits[i] for i in course_ids if x_vars[i].X > 0.5))
    print(f'Optimal total interest points: {m.objVal:.0f}')
    print(f'Total credits used: {total_credits}')
    print('--- Selected Operations Research Courses ---')
    for i in course_ids:
        if x_vars[i].X > 0.5:
            row = ops_research_df[ops_research_df['course_id'] == i].iloc[0]
            print(f"  {row['course_id']}: {row['course_name']} | Credits: {row['credits']} | Interest Points: {row['interest_points']}")
else:
    print(f'No optimal solution found. Status: {m.status}')