import gurobipy as gp
import pandas as pd
import numpy as np
school_cap_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv', sep=',')
schools = [str(s).strip() for s in school_cap_df['School']]
school_capacities = {str(row['School']).strip(): int(row['Capacity']) for _, row in school_cap_df.iterrows()}
neigh_pop_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv', sep=',')
neighborhoods = [str(n).strip() for n in neigh_pop_df['Neighborhood']]
groups = ['White', 'NonWhite']
pop_white = {str(row['Neighborhood']).strip(): int(row['Population_White']) for _, row in neigh_pop_df.iterrows()}
pop_nonwhite = {str(row['Neighborhood']).strip(): int(row['Population_NonWhite']) for _, row in neigh_pop_df.iterrows()}
pop_by_group = {'White': pop_white, 'NonWhite': pop_nonwhite}
dist_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv', sep=',')
dist_df['School'] = dist_df['School'].astype(str).str.strip()
distance = {}
for _, row in dist_df.iterrows():
    s = str(row['School']).strip()
    distance[s] = {}
    for n in neighborhoods:
        distance[s][n] = float(row[n])
if set(schools) != set(distance.keys()):
    raise ValueError('Mismatch between schools in school_capacity.csv and distance.csv')
if not all((n in pop_white and n in pop_nonwhite for n in neighborhoods)):
    raise ValueError('Mismatch in neighborhoods between population and distance files')
for s in schools:
    for n in neighborhoods:
        if n not in distance[s]:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighborhoods, groups, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[s][n] * x[s, n, g] for s in schools for n in neighborhoods for g in groups)), gp.GRB.MINIMIZE)
for n in neighborhoods:
    for g in groups:
        m.addConstr(gp.quicksum((x[s, n, g] for s in schools)) == pop_by_group[g][n])
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups)) <= school_capacities[s])
for s in schools:
    total_white = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
    total_students = gp.quicksum((x[s, n, g] for n in neighborhoods for g in groups))
    m.addConstr(total_white >= 0.5 * total_students)
    m.addConstr(total_white <= 0.7 * total_students)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} miles')
    print('\nAssignment by school, neighborhood, and group (nonzero only):')
    for s in schools:
        for n in neighborhoods:
            for g in groups:
                val = x[s, n, g].X
                if val > 1e-06:
                    print(f'School {s}, Neighborhood {n}, Group {g}: {val:.2f} students')
    print('\nSchool-level summary:')
    for s in schools:
        total_white = sum((x[s, n, 'White'].X for n in neighborhoods))
        total_nonwhite = sum((x[s, n, 'NonWhite'].X for n in neighborhoods))
        total_students = total_white + total_nonwhite
        pct_white = total_white / total_students if total_students > 0 else 0.0
        print(f'School {s}: {total_students:.0f} students assigned (White: {total_white:.0f}, NonWhite: {total_nonwhite:.0f}), White %: {pct_white * 100:.2f}%')
else:
    print(f'No optimal solution found. Status: {m.status}')