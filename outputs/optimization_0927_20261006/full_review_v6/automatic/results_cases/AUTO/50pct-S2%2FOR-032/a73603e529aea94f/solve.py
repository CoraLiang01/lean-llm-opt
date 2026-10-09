import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_df = df[ops_research_mask].copy()
if ops_research_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
ops_course_ids = ops_research_df['course_id'].tolist()
try:
    credits = dict(zip(ops_research_df['course_id'], ops_research_df['credits'].astype(int)))
    interest_points = dict(zip(ops_research_df['course_id'], ops_research_df['interest_points'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(ops_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in ops_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in ops_course_ids)) <= 20, name='TotalCredits')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in ops_course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total value/cost: {m.objVal} (Total interest points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for i in selected:
        row = ops_research_df[ops_research_df['course_id'] == i].iloc[0]
        print(f"  {row['course_id']}: {row['course_name']} | Credits: {credits[i]} | Interest Points: {interest_points[i]}")
    print(f'\nTotal selected: {len(selected)} courses')
else:
    print(f'No optimal solution found. Status: {m.status}')