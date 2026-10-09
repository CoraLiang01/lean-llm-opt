CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Educational planners must be able to address the travel-time issue, while striving to satisfy other '
          'requirements. One approach to the problem is to consider racial balance, school capacity, and population of '
          'the school district as constraints, while striving to transport the children with as little inconvenience '
          'to them as possible. Suppose the Capital City School District has two elementary schools, with individual '
          'capacities in school_capacity.csv, and 31 neighborhoods with elementary school populations summarized in '
          'neighborhoods_population.csv. The distance in miles from each school to each neighborhood is given in '
          'distance.csv. The district has decided to integrate by school rather than by grade level. A school will be '
          'considered to be in racial balance when its white-student percentage deviates by no more than 10 percentage '
          'points from the district ratio of 60 percent white and 40 percent nonwhite. The district wishes to devise a '
          'busing plan that assigns white and nonwhite pupils from each neighborhood to schools, respects school '
          'capacities and neighborhood populations, and minimizes the total distance the youngsters must travel.',
 'relationships': [{'column_axis': {'id_column': 'Neighborhood', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'School', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'School',
                    'type': 'matrix'}],
 'route': 'Others',
 'tables': [{'columns': ['School', 'Capacity'],
             'file_index': 0,
             'file_name': 'school_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Capacity': '2028', 'School': 'I'}},
                         {'source_row': 1, 'values': {'Capacity': '1560', 'School': 'II'}}],
             'returned_rows': 2,
             'role': 'school capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['Neighborhood', 'Population_White', 'Population_NonWhite'],
             'file_index': 1,
             'file_name': 'neighborhoods_population.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 31,
             'records': [{'source_row': 0,
                          'values': {'Neighborhood': 'N01', 'Population_NonWhite': '22', 'Population_White': '78'}},
                         {'source_row': 1,
                          'values': {'Neighborhood': 'N02', 'Population_NonWhite': '33', 'Population_White': '57'}},
                         {'source_row': 2,
                          'values': {'Neighborhood': 'N03', 'Population_NonWhite': '63', 'Population_White': '47'}},
                         {'source_row': 3,
                          'values': {'Neighborhood': 'N04', 'Population_NonWhite': '22', 'Population_White': '78'}},
                         {'source_row': 4,
                          'values': {'Neighborhood': 'N05', 'Population_NonWhite': '33', 'Population_White': '57'}},
                         {'source_row': 5,
                          'values': {'Neighborhood': 'N06', 'Population_NonWhite': '63', 'Population_White': '47'}},
                         {'source_row': 6,
                          'values': {'Neighborhood': 'N07', 'Population_NonWhite': '22', 'Population_White': '78'}},
                         {'source_row': 7,
                          'values': {'Neighborhood': 'N08', 'Population_NonWhite': '33', 'Population_White': '57'}},
                         {'source_row': 8,
                          'values': {'Neighborhood': 'N09', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 9,
                          'values': {'Neighborhood': 'N10', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 10,
                          'values': {'Neighborhood': 'N11', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 11,
                          'values': {'Neighborhood': 'N12', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 12,
                          'values': {'Neighborhood': 'N13', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 13,
                          'values': {'Neighborhood': 'N14', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 14,
                          'values': {'Neighborhood': 'N15', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 15,
                          'values': {'Neighborhood': 'N16', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 16,
                          'values': {'Neighborhood': 'N17', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 17,
                          'values': {'Neighborhood': 'N18', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 18,
                          'values': {'Neighborhood': 'N19', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 19,
                          'values': {'Neighborhood': 'N20', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 20,
                          'values': {'Neighborhood': 'N21', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 21,
                          'values': {'Neighborhood': 'N22', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 22,
                          'values': {'Neighborhood': 'N23', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 23,
                          'values': {'Neighborhood': 'N24', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 24,
                          'values': {'Neighborhood': 'N25', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 25,
                          'values': {'Neighborhood': 'N26', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 26,
                          'values': {'Neighborhood': 'N27', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 27,
                          'values': {'Neighborhood': 'N28', 'Population_NonWhite': '23', 'Population_White': '77'}},
                         {'source_row': 28,
                          'values': {'Neighborhood': 'N29', 'Population_NonWhite': '34', 'Population_White': '56'}},
                         {'source_row': 29,
                          'values': {'Neighborhood': 'N30', 'Population_NonWhite': '64', 'Population_White': '46'}},
                         {'source_row': 30,
                          'values': {'Neighborhood': 'N31', 'Population_NonWhite': '46', 'Population_White': '74'}}],
             'returned_rows': 31,
             'role': 'neighborhood population',
             'table_id': 'file_1_view_0'},
            {'columns': ['School',
                         'N01',
                         'N02',
                         'N03',
                         'N04',
                         'N05',
                         'N06',
                         'N07',
                         'N08',
                         'N09',
                         'N10',
                         'N11',
                         'N12',
                         'N13',
                         'N14',
                         'N15',
                         'N16',
                         'N17',
                         'N18',
                         'N19',
                         'N20',
                         'N21',
                         'N22',
                         'N23',
                         'N24',
                         'N25',
                         'N26',
                         'N27',
                         'N28',
                         'N29',
                         'N30',
                         'N31'],
             'file_index': 2,
             'file_name': 'distance.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0,
                          'values': {'N01': '1.25',
                                     'N02': '1.3',
                                     'N03': '1.35',
                                     'N04': '1.4',
                                     'N05': '1.45',
                                     'N06': '1.5',
                                     'N07': '1.55',
                                     'N08': '1.6',
                                     'N09': '1.65',
                                     'N10': '1.7',
                                     'N11': '1.75',
                                     'N12': '1.8',
                                     'N13': '1.85',
                                     'N14': '1.9',
                                     'N15': '1.95',
                                     'N16': '2.0',
                                     'N17': '3.08',
                                     'N18': '3.16',
                                     'N19': '3.24',
                                     'N20': '3.32',
                                     'N21': '3.4',
                                     'N22': '3.48',
                                     'N23': '3.56',
                                     'N24': '3.64',
                                     'N25': '3.7199999999999998',
                                     'N26': '3.8',
                                     'N27': '3.88',
                                     'N28': '3.96',
                                     'N29': '4.04',
                                     'N30': '4.12',
                                     'N31': '4.2',
                                     'School': 'I'}},
                         {'source_row': 1,
                          'values': {'N01': '3.08',
                                     'N02': '3.16',
                                     'N03': '3.24',
                                     'N04': '3.32',
                                     'N05': '3.4',
                                     'N06': '3.48',
                                     'N07': '3.56',
                                     'N08': '3.64',
                                     'N09': '3.7199999999999998',
                                     'N10': '3.8',
                                     'N11': '3.88',
                                     'N12': '3.96',
                                     'N13': '4.04',
                                     'N14': '4.12',
                                     'N15': '4.2',
                                     'N16': '4.28',
                                     'N17': '1.25',
                                     'N18': '1.3',
                                     'N19': '1.35',
                                     'N20': '1.4',
                                     'N21': '1.45',
                                     'N22': '1.5',
                                     'N23': '1.55',
                                     'N24': '1.6',
                                     'N25': '1.65',
                                     'N26': '1.7',
                                     'N27': '1.75',
                                     'N28': '1.8',
                                     'N29': '1.85',
                                     'N30': '1.9',
                                     'N31': '1.95',
                                     'School': 'II'}}],
             'returned_rows': 2,
             'role': 'school-neighborhood distance matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [2, 31],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [2, 31]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem(CSVQA_FRAMES):
    school_frame = CSVQA_FRAMES['file_0_view_0']
    pop_frame = CSVQA_FRAMES['file_1_view_0']
    dist_frame = CSVQA_FRAMES['file_2_view_0']
    S = [row['School'] for (_, row) in school_frame.iterrows()]
    N = [row['Neighborhood'] for (_, row) in pop_frame.iterrows()]
    R = ['White', 'NonWhite']
    C_s = {}
    for (_, row) in school_frame.iterrows():
        C_s[row['School']] = float(row['Capacity'])
    P_n_r = {}
    for (_, row) in pop_frame.iterrows():
        n = row['Neighborhood']
        P_n_r[n, 'White'] = float(row['Population_White'])
        P_n_r[n, 'NonWhite'] = float(row['Population_NonWhite'])
    d_s_n = {}
    for (_, row) in dist_frame.iterrows():
        s = row['School']
        for n in N:
            d_s_n[s, n] = float(row[n])
    P_total_r = {}
    for r in R:
        P_total_r[r] = sum((P_n_r[n, r] for n in N))
    P_total = sum((P_total_r[r] for r in R))
    alpha_white = P_total_r['White'] / P_total if P_total > 0 else 0.0
    epsilon = 0.1
    for s in S:
        if s not in C_s:
            raise ValueError(f'Missing capacity for school {s}')
    for n in N:
        for r in R:
            if (n, r) not in P_n_r:
                raise ValueError(f'Missing population for neighborhood {n}, race {r}')
    for s in S:
        for n in N:
            if (s, n) not in d_s_n:
                raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
    m = gp.Model('SchoolAssignment')
    keys = [(s, n, r) for s in S for n in N for r in R]
    x_vars = m.addVars(keys, lb=0.0, name='')
    m.setObjective(gp.quicksum((d_s_n[s, n] * x_vars[s, n, r] for s in S for n in N for r in R)), gp.GRB.MINIMIZE)
    for n in N:
        for r in R:
            m.addConstr(gp.quicksum((x_vars[s, n, r] for s in S)) == P_n_r[n, r], name=f'Assign_{n}_{r}')
    for s in S:
        m.addConstr(gp.quicksum((x_vars[s, n, r] for n in N for r in R)) <= C_s[s], name=f'Capacity_{s}')
    lower = alpha_white - epsilon
    upper = alpha_white + epsilon
    for s in S:
        total_students = gp.quicksum((x_vars[s, n, r] for n in N for r in R))
        white_students = gp.quicksum((x_vars[s, n, 'White'] for n in N))
        m.addConstr(white_students >= lower * total_students, name=f'RacialBalanceLower_{s}')
        m.addConstr(white_students <= upper * total_students, name=f'RacialBalanceUpper_{s}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)