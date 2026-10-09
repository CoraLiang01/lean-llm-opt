CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A water authority must install monitoring sensors so every monitoring zone is covered. Candidate sensor '
          'costs and covered zones are listed in sensor_sites.csv, and the complete set of monitoring zones is listed '
          'in monitoring_zones.csv.\n'
          '\n'
          'Formulate a minimum-cost set covering model. For each candidate sensor site i, define y_i as a binary '
          'variable equal to 1 if a sensor is installed at site i. The objective is to minimize total installation '
          'cost. The model should include one coverage constraint for each monitoring zone requiring at least one '
          'selected site that covers that zone, and binary restrictions for all sensor-site variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Center', 'OpeningCost', 'CoveredDistricts'],
             'file_index': 0,
             'file_name': 'sensor_sites.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0,
                          'values': {'Center': 'S1', 'CoveredDistricts': 'Q1;Q2;Q4', 'OpeningCost': '8'}},
                         {'source_row': 1,
                          'values': {'Center': 'S2', 'CoveredDistricts': 'Q2;Q3;Q5', 'OpeningCost': '12'}},
                         {'source_row': 2,
                          'values': {'Center': 'S3', 'CoveredDistricts': 'Q4;Q6;Q7', 'OpeningCost': '11'}},
                         {'source_row': 3,
                          'values': {'Center': 'S4', 'CoveredDistricts': 'Q5;Q8', 'OpeningCost': '10'}},
                         {'source_row': 4,
                          'values': {'Center': 'S5', 'CoveredDistricts': 'Q6;Q8;Q9', 'OpeningCost': '13'}},
                         {'source_row': 5,
                          'values': {'Center': 'S6', 'CoveredDistricts': 'Q7;Q10;Q11', 'OpeningCost': '9'}},
                         {'source_row': 6,
                          'values': {'Center': 'S7', 'CoveredDistricts': 'Q1;Q9;Q10', 'OpeningCost': '14'}},
                         {'source_row': 7,
                          'values': {'Center': 'S8', 'CoveredDistricts': 'Q3;Q5;Q11', 'OpeningCost': '7'}},
                         {'source_row': 8,
                          'values': {'Center': 'S9', 'CoveredDistricts': 'Q2;Q6;Q10', 'OpeningCost': '15'}}],
             'returned_rows': 9,
             'role': 'candidate sensor sites and coverage',
             'table_id': 'file_0_view_0'},
            {'columns': ['Zone'],
             'file_index': 1,
             'file_name': 'monitoring_zones.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0, 'values': {'Zone': 'Q1'}},
                         {'source_row': 1, 'values': {'Zone': 'Q2'}},
                         {'source_row': 2, 'values': {'Zone': 'Q3'}},
                         {'source_row': 3, 'values': {'Zone': 'Q4'}},
                         {'source_row': 4, 'values': {'Zone': 'Q5'}},
                         {'source_row': 5, 'values': {'Zone': 'Q6'}},
                         {'source_row': 6, 'values': {'Zone': 'Q7'}},
                         {'source_row': 7, 'values': {'Zone': 'Q8'}},
                         {'source_row': 8, 'values': {'Zone': 'Q9'}},
                         {'source_row': 9, 'values': {'Zone': 'Q10'}},
                         {'source_row': 10, 'values': {'Zone': 'Q11'}}],
             'returned_rows': 11,
             'role': 'monitoring zones',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    sensor_sites_df = CSVQA_FRAMES['file_0_view_0']
    monitoring_zones_df = CSVQA_FRAMES['file_1_view_0']
    I = []
    C = {}
    covered_zones = {}
    for (idx, row) in sensor_sites_df.iterrows():
        site = row['Center']
        I.append(site)
        try:
            C[site] = float(row['OpeningCost'])
        except Exception:
            raise ValueError(f"Non-numeric OpeningCost for site {site}: {row['OpeningCost']}")
        if pd.isna(row['CoveredDistricts']) or row['CoveredDistricts'] == '':
            covered_zones[site] = set()
        else:
            covered_zones[site] = set((z.strip() for z in row['CoveredDistricts'].split(';') if z.strip()))
    J = []
    for (idx, row) in monitoring_zones_df.iterrows():
        zone = row['Zone']
        J.append(zone)
    zone_covered_by = {j: [] for j in J}
    for i in I:
        for j in covered_zones[i]:
            if j in zone_covered_by:
                zone_covered_by[j].append(i)
    for j in J:
        if not zone_covered_by[j]:
            raise ValueError(f'Monitoring zone {j} is not covered by any sensor site.')
    A = {}
    for i in I:
        for j in J:
            A[i, j] = 1 if j in covered_zones[i] else 0
    m = gp.Model('SensorSetCover')
    y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((C[i] * y_vars[i] for i in I)), gp.GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((A[i, j] * y_vars[i] for i in I)) >= 1, name=f'cover_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)