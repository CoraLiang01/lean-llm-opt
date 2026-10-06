import gurobipy as gp
import pandas as pd
import numpy as np
school_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'

def solve_school_assignment():
    school_df = pd.read_csv(school_capacity_path, sep=',')
    neighborhoods_df = pd.read_csv(neighborhoods_population_path, sep=',')
    distance_df = pd.read_csv(distance_path, sep=',')
    schools = [str(s).strip() for s in school_df['School']]
    neighborhoods = [str(n).strip() for n in neighborhoods_df['Neighborhood']]
    groups = ['White', 'NonWhite']
    school_capacity = {str(row['School']).strip(): int(row['Capacity']) for _, row in school_df.iterrows()}
    pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for _, row in neighborhoods_df.iterrows()}
    pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for _, row in neighborhoods_df.iterrows()}
    pop_by_group = {'White': pop_white, 'NonWhite': pop_nonwhite}
    total_white = sum((pop_white[n] for n in neighborhoods))
    total_nonwhite = sum((pop_nonwhite[n] for n in neighborhoods))
    total_students = total_white + total_nonwhite
    distance = {}
    for _, row in distance_df.iterrows():
        school = str(row['School']).strip()
        for n in neighborhoods:
            distance[school, n] = float(row[n])
    m = gp.Model('SchoolAssignment')
    x = m.addVars(schools, neighborhoods, groups, lb=0.0, ub=lambda s, n, g: pop_by_group[g][n], vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((distance[s, n] * x[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
    for n in neighborhoods:
        for g in groups:
            m.addConstr(gp.quicksum((x[s, n, g] for s in schools)) == pop_by_group[g][n], name=f'assign_{n}_{g}')
    for s in schools:
        m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacity[s], name=f'capacity_{s}')
    for s in schools:
        total_assigned = gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups))
        total_white = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
        m.addConstr(total_white >= 0.5 * total_assigned, name=f'racial_lb_{s}')
        m.addConstr(total_white <= 0.7 * total_assigned, name=f'racial_ub_{s}')
    m.optimize()
    return m
m = solve_school_assignment()