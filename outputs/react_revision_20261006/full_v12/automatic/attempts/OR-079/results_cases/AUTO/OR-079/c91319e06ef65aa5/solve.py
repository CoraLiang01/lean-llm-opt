CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'ElectroTech Manufacturing is undergoing a critical strategic network restructuring. The primary objective '
          'is to minimize the total system cost by optimizing the supply chain backbone. This requires a dual '
          'decision: 1) Determine which subset of 15 potential factory sites (A1-A15) to construct (considering fixed '
          'costs from facility_costs.csv). 2) Plan the most economical shipment routes from these selected factories '
          'to satisfy the fixed demand of 8 distribution centers (B1-B8) (demand from demand_requirements.csv; '
          'variable costs from shipping_costs.csv).The core task is to find the optimal balance between fixed '
          'investment and variable logistics costs while ensuring all market demand is met.',
 'relationships': [{'column_axis': {'id_column': 'Destination', 'table_id': 'file_2_view_0'},
                    'matrix_table_id': 'file_1_view_0',
                    'row_axis': {'id_column': 'Facility', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Origin',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['Facility', 'FixedCost', 'Capacity'],
             'file_index': 0,
             'file_name': 'facility_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'Capacity': '30', 'Facility': 'A1', 'FixedCost': '0'}},
                         {'source_row': 1, 'values': {'Capacity': '10', 'Facility': 'A2', 'FixedCost': '175'}},
                         {'source_row': 2, 'values': {'Capacity': '20', 'Facility': 'A3', 'FixedCost': '300'}},
                         {'source_row': 3, 'values': {'Capacity': '30', 'Facility': 'A4', 'FixedCost': '375'}},
                         {'source_row': 4, 'values': {'Capacity': '40', 'Facility': 'A5', 'FixedCost': '500'}},
                         {'source_row': 5, 'values': {'Capacity': '20', 'Facility': 'A6', 'FixedCost': '200'}},
                         {'source_row': 6, 'values': {'Capacity': '25', 'Facility': 'A7', 'FixedCost': '260'}},
                         {'source_row': 7, 'values': {'Capacity': '30', 'Facility': 'A8', 'FixedCost': '220'}},
                         {'source_row': 8, 'values': {'Capacity': '35', 'Facility': 'A9', 'FixedCost': '320'}},
                         {'source_row': 9, 'values': {'Capacity': '20', 'Facility': 'A10', 'FixedCost': '280'}},
                         {'source_row': 10, 'values': {'Capacity': '40', 'Facility': 'A11', 'FixedCost': '350'}},
                         {'source_row': 11, 'values': {'Capacity': '25', 'Facility': 'A12', 'FixedCost': '420'}},
                         {'source_row': 12, 'values': {'Capacity': '30', 'Facility': 'A13', 'FixedCost': '470'}},
                         {'source_row': 13, 'values': {'Capacity': '50', 'Facility': 'A14', 'FixedCost': '520'}},
                         {'source_row': 14, 'values': {'Capacity': '45', 'Facility': 'A15', 'FixedCost': '560'}}],
             'returned_rows': 15,
             'role': 'facility fixed costs and capacities',
             'table_id': 'file_0_view_0'},
            {'columns': ['Origin', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
             'file_index': 1,
             'file_name': 'shipping_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'B1': '8',
                                     'B2': '4',
                                     'B3': '3',
                                     'B4': '6',
                                     'B5': '7',
                                     'B6': '5',
                                     'B7': '9',
                                     'B8': '8',
                                     'Origin': 'A1'}},
                         {'source_row': 1,
                          'values': {'B1': '5',
                                     'B2': '2',
                                     'B3': '3',
                                     'B4': '5',
                                     'B5': '6',
                                     'B6': '4',
                                     'B7': '7',
                                     'B8': '6',
                                     'Origin': 'A2'}},
                         {'source_row': 2,
                          'values': {'B1': '4',
                                     'B2': '3',
                                     'B3': '4',
                                     'B4': '6',
                                     'B5': '5',
                                     'B6': '5',
                                     'B7': '6',
                                     'B8': '7',
                                     'Origin': 'A3'}},
                         {'source_row': 3,
                          'values': {'B1': '9',
                                     'B2': '7',
                                     'B3': '5',
                                     'B4': '8',
                                     'B5': '9',
                                     'B6': '6',
                                     'B7': '10',
                                     'B8': '7',
                                     'Origin': 'A4'}},
                         {'source_row': 4,
                          'values': {'B1': '10',
                                     'B2': '4',
                                     'B3': '2',
                                     'B4': '6',
                                     'B5': '8',
                                     'B6': '5',
                                     'B7': '7',
                                     'B8': '3',
                                     'Origin': 'A5'}},
                         {'source_row': 5,
                          'values': {'B1': '6',
                                     'B2': '5',
                                     'B3': '4',
                                     'B4': '5',
                                     'B5': '7',
                                     'B6': '6',
                                     'B7': '8',
                                     'B8': '5',
                                     'Origin': 'A6'}},
                         {'source_row': 6,
                          'values': {'B1': '7',
                                     'B2': '6',
                                     'B3': '5',
                                     'B4': '4',
                                     'B5': '6',
                                     'B6': '7',
                                     'B7': '9',
                                     'B8': '6',
                                     'Origin': 'A7'}},
                         {'source_row': 7,
                          'values': {'B1': '5',
                                     'B2': '4',
                                     'B3': '6',
                                     'B4': '3',
                                     'B5': '5',
                                     'B6': '6',
                                     'B7': '7',
                                     'B8': '6',
                                     'Origin': 'A8'}},
                         {'source_row': 8,
                          'values': {'B1': '8',
                                     'B2': '7',
                                     'B3': '6',
                                     'B4': '7',
                                     'B5': '9',
                                     'B6': '8',
                                     'B7': '10',
                                     'B8': '7',
                                     'Origin': 'A9'}},
                         {'source_row': 9,
                          'values': {'B1': '6',
                                     'B2': '5',
                                     'B3': '7',
                                     'B4': '4',
                                     'B5': '6',
                                     'B6': '5',
                                     'B7': '7',
                                     'B8': '5',
                                     'Origin': 'A10'}},
                         {'source_row': 10,
                          'values': {'B1': '9',
                                     'B2': '6',
                                     'B3': '4',
                                     'B4': '6',
                                     'B5': '8',
                                     'B6': '7',
                                     'B7': '9',
                                     'B8': '6',
                                     'Origin': 'A11'}},
                         {'source_row': 11,
                          'values': {'B1': '7',
                                     'B2': '5',
                                     'B3': '6',
                                     'B4': '5',
                                     'B5': '6',
                                     'B6': '5',
                                     'B7': '8',
                                     'B8': '5',
                                     'Origin': 'A12'}},
                         {'source_row': 12,
                          'values': {'B1': '8',
                                     'B2': '6',
                                     'B3': '5',
                                     'B4': '6',
                                     'B5': '7',
                                     'B6': '6',
                                     'B7': '8',
                                     'B8': '7',
                                     'Origin': 'A13'}},
                         {'source_row': 13,
                          'values': {'B1': '9',
                                     'B2': '5',
                                     'B3': '3',
                                     'B4': '5',
                                     'B5': '7',
                                     'B6': '4',
                                     'B7': '6',
                                     'B8': '4',
                                     'Origin': 'A14'}},
                         {'source_row': 14,
                          'values': {'B1': '10',
                                     'B2': '6',
                                     'B3': '4',
                                     'B4': '5',
                                     'B5': '8',
                                     'B6': '5',
                                     'B7': '7',
                                     'B8': '5',
                                     'Origin': 'A15'}}],
             'returned_rows': 15,
             'role': 'factory-to-distribution center shipping costs matrix',
             'table_id': 'file_1_view_0'},
            {'columns': ['Destination', 'Demand'],
             'file_index': 2,
             'file_name': 'demand_requirements.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Demand': '30', 'Destination': 'B1'}},
                         {'source_row': 1, 'values': {'Demand': '25', 'Destination': 'B2'}},
                         {'source_row': 2, 'values': {'Demand': '20', 'Destination': 'B3'}},
                         {'source_row': 3, 'values': {'Demand': '35', 'Destination': 'B4'}},
                         {'source_row': 4, 'values': {'Demand': '25', 'Destination': 'B5'}},
                         {'source_row': 5, 'values': {'Demand': '30', 'Destination': 'B6'}},
                         {'source_row': 6, 'values': {'Demand': '25', 'Destination': 'B7'}},
                         {'source_row': 7, 'values': {'Demand': '30', 'Destination': 'B8'}}],
             'returned_rows': 8,
             'role': 'distribution center demand requirements',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [15, 8],
                                   'matrix_table_id': 'file_1_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [15, 8]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    facility_frame = CSVQA_FRAMES['file_0_view_0']
    facilities = []
    F = {}
    K = {}
    for (_, row) in facility_frame.iterrows():
        fac = row['Facility']
        facilities.append(fac)
        try:
            F[fac] = float(row['FixedCost'])
        except Exception:
            raise ValueError(f"Invalid FixedCost for facility {fac}: {row['FixedCost']}")
        try:
            K[fac] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for facility {fac}: {row['Capacity']}")
    demand_frame = CSVQA_FRAMES['file_2_view_0']
    dcs = []
    d = {}
    for (_, row) in demand_frame.iterrows():
        dc = row['Destination']
        dcs.append(dc)
        try:
            d[dc] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand for destination {dc}: {row['Demand']}")
    shipping_frame = CSVQA_FRAMES['file_1_view_0']
    c = {}
    for (_, row) in shipping_frame.iterrows():
        i = row['Origin']
        c[i] = {}
        for j in dcs:
            try:
                c[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid shipping cost from {i} to {j}: {row[j]}')
    if set(facilities) != set(c.keys()):
        raise ValueError('Mismatch between facilities in facility_costs.csv and shipping_costs.csv')
    for i in facilities:
        if set(dcs) != set(c[i].keys()):
            raise ValueError(f'Mismatch in shipping cost columns for facility {i}')
    m = gp.Model('ElectroTech_FLP')
    m.setParam('MIPGap', 0.0001)
    y_vars = m.addVars(facilities, vtype=GRB.BINARY, name='')
    x_vars = m.addVars([(i, j) for i in facilities for j in dcs], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((F[i] * y_vars[i] for i in facilities)) + gp.quicksum((c[i][j] * x_vars[i, j] for i in facilities for j in dcs)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in facilities)) == d[j] for j in dcs), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in dcs)) <= K[i] * y_vars[i] for i in facilities), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')