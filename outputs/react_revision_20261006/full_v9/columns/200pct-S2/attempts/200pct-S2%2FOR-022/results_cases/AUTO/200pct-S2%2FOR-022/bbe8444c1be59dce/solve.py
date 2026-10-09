CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the ‚Äúfixed_cost.csv‚Äù file. Each branch needs to source '
          'a certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of '
          'goods from each supplier is recorded in the ‚Äútransportation_costs.csv‚Äù file. Demand information can be '
          "gained in 'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all "
          'branches is met while minimizing the total cost. The decision variables y_i are binary, indicating whether '
          'a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_C1': 'C1',
                                          'transportation_cost_to_C2': 'C2',
                                          'transportation_cost_to_C3': 'C3',
                                          'transportation_cost_to_C4': 'C4',
                                          'transportation_cost_to_C5': 'C5'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'facility_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'facility_id',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer_id', 'demand_units'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand_units': '143'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand_units': '6'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand_units': '10'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand_units': '25'}},
                         {'source_row': 4, 'values': {'customer_id': 'C5', 'demand_units': '3'}}],
             'returned_rows': 5,
             'role': 'branch demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['facility_id', 'fixed_opening_cost'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'facility_id': 'S1', 'fixed_opening_cost': '97.65'}},
                         {'source_row': 1, 'values': {'facility_id': 'S2', 'fixed_opening_cost': '99.76'}},
                         {'source_row': 2, 'values': {'facility_id': 'S3', 'fixed_opening_cost': '100.76'}},
                         {'source_row': 3, 'values': {'facility_id': 'S4', 'fixed_opening_cost': '105.32'}},
                         {'source_row': 4, 'values': {'facility_id': 'S5', 'fixed_opening_cost': '98.88'}}],
             'returned_rows': 5,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['facility_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'facility_id': 'S1',
                                     'transportation_cost_to_C1': '150.74',
                                     'transportation_cost_to_C2': '0.02',
                                     'transportation_cost_to_C3': '49.13',
                                     'transportation_cost_to_C4': '2080.15',
                                     'transportation_cost_to_C5': '426.4'}},
                         {'source_row': 1,
                          'values': {'facility_id': 'S2',
                                     'transportation_cost_to_C1': '233.05',
                                     'transportation_cost_to_C2': '97.73',
                                     'transportation_cost_to_C3': '49.84',
                                     'transportation_cost_to_C4': '1982.39',
                                     'transportation_cost_to_C5': '23.96'}},
                         {'source_row': 2,
                          'values': {'facility_id': 'S3',
                                     'transportation_cost_to_C1': '55.68',
                                     'transportation_cost_to_C2': '935.61',
                                     'transportation_cost_to_C3': '4.03',
                                     'transportation_cost_to_C4': '73.09',
                                     'transportation_cost_to_C5': '525.32'}},
                         {'source_row': 3,
                          'values': {'facility_id': 'S4',
                                     'transportation_cost_to_C1': '1483.82',
                                     'transportation_cost_to_C2': '1801.08',
                                     'transportation_cost_to_C3': '112.16',
                                     'transportation_cost_to_C4': '816.05',
                                     'transportation_cost_to_C5': '107.01'}},
                         {'source_row': 4,
                          'values': {'facility_id': 'S5',
                                     'transportation_cost_to_C1': '1119.47',
                                     'transportation_cost_to_C2': '884.31',
                                     'transportation_cost_to_C3': '0.08',
                                     'transportation_cost_to_C4': '1544.95',
                                     'transportation_cost_to_C5': '543.67'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    transport_cost_frame = CSVQA_FRAMES['file_2_view_0']
    branches = []
    for (_, row) in demand_frame.iterrows():
        cid = row['customer_id']
        if cid not in branches:
            branches.append(cid)
    for col in transport_cost_frame.columns:
        if col.startswith('transportation_cost_to_'):
            branch_id = col[len('transportation_cost_to_'):]
            if branch_id not in branches:
                branches.append(branch_id)
    suppliers = []
    for (_, row) in fixed_cost_frame.iterrows():
        fid = row['facility_id']
        if fid not in suppliers:
            suppliers.append(fid)
    for (_, row) in transport_cost_frame.iterrows():
        fid = row['facility_id']
        if fid not in suppliers:
            suppliers.append(fid)
    demand = {}
    for (_, row) in demand_frame.iterrows():
        cid = row['customer_id']
        val = row['demand_units']
        try:
            demand[cid] = float(val)
        except Exception:
            raise ValueError(f'Invalid demand_units for customer_id {cid}: {val}')
    fixed_cost = {}
    for (_, row) in fixed_cost_frame.iterrows():
        fid = row['facility_id']
        val = row['fixed_opening_cost']
        try:
            fixed_cost[fid] = float(val)
        except Exception:
            raise ValueError(f'Invalid fixed_opening_cost for facility_id {fid}: {val}')
    cost = {}
    for (_, row) in transport_cost_frame.iterrows():
        fid = row['facility_id']
        if fid not in cost:
            cost[fid] = {}
        for j in branches:
            col = f'transportation_cost_to_{j}'
            if col in transport_cost_frame.columns:
                val = row[col]
                try:
                    cost[fid][j] = float(val)
                except Exception:
                    raise ValueError(f'Invalid transportation_cost_to_{j} for facility_id {fid}: {val}')
            else:
                raise ValueError(f'Missing transportation_cost_to_{j} for facility_id {fid}')
    M = sum((demand[j] for j in branches))
    M_i = {i: M for i in suppliers}
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed_opening_cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in branches:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, branch {j}')
    for j in branches:
        if j not in demand:
            raise ValueError(f'Missing demand for branch {j}')
    m = gp.Model('FLP')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(suppliers, branches, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in branches)) + gp.quicksum((fixed_cost[i] * open_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j] for j in branches), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in branches)) <= M_i[i] * open_vars[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()