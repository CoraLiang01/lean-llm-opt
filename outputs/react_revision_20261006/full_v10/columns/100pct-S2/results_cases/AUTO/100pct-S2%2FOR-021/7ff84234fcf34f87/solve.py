CSVQA_DATA = {'ignored_file_indices': [],
 'query': '‚ÄúBrewCo,‚Äù a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in '
          '‚Äúcustomer_demand.csv,‚Äù while the production capacity of each plant is outlined in '
          '‚Äúsupply_capacity.csv.‚Äù The transportation cost per unit of beverages from each plant to each outlet is '
          'recorded in ‚Äútransportation_costs.csv.‚Äù The goal is to determine the optimal quantity of beverages to '
          'be shipped from each production plant to each retail outlet, ensuring all outlet demands are met without '
          'surpassing any plant‚Äôs production capacity, while minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_C1': 'C1',
                                          'transportation_cost_to_C2': 'C2',
                                          'transportation_cost_to_C3': 'C3',
                                          'transportation_cost_to_C4': 'C4'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'supplier_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'supplier_id',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand': '94'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand': '39'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand': '65'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'supplier_id': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1, 'values': {'supplier_id': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2, 'values': {'supplier_id': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3, 'values': {'supplier_id': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'plant supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'supplier_id': 'S1',
                                     'transportation_cost_to_C1': '543.756480860856',
                                     'transportation_cost_to_C2': '23.685276141764653',
                                     'transportation_cost_to_C3': '23.676386730773032',
                                     'transportation_cost_to_C4': '447.75143678673766'}},
                         {'source_row': 1,
                          'values': {'supplier_id': 'S2',
                                     'transportation_cost_to_C1': '883.9151090405642',
                                     'transportation_cost_to_C2': '0.04977684765576961',
                                     'transportation_cost_to_C3': '0.0350986687216299',
                                     'transportation_cost_to_C4': '44.45588531711622'}},
                         {'source_row': 2,
                          'values': {'supplier_id': 'S3',
                                     'transportation_cost_to_C1': '537.3456896658107',
                                     'transportation_cost_to_C2': '23.769274659075112',
                                     'transportation_cost_to_C3': '498.95659249465467',
                                     'transportation_cost_to_C4': '440.60737890439776'}},
                         {'source_row': 3,
                          'values': {'supplier_id': 'S4',
                                     'transportation_cost_to_C1': '1791.493192397229',
                                     'transportation_cost_to_C2': '68.21633865655126',
                                     'transportation_cost_to_C3': '1432.4837339656747',
                                     'transportation_cost_to_C4': '1527.7635425462734'}}],
             'returned_rows': 4,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 4]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    import pandas as pd
    I = []
    J = []
    d_j = {}
    s_i = {}
    c_ij = {}
    frame_demand = CSVQA_FRAMES['file_0_view_0']
    for (source_row, row) in frame_demand.iterrows():
        customer_id = row['customer_id']
        J.append(customer_id)
        try:
            d_j[customer_id] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {customer_id}: {row['demand']}")
    frame_supply = CSVQA_FRAMES['file_1_view_0']
    for (source_row, row) in frame_supply.iterrows():
        supplier_id = row['supplier_id']
        I.append(supplier_id)
        try:
            s_i[supplier_id] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for supplier {supplier_id}: {row['supply_capacity']}")
    frame_cost = CSVQA_FRAMES['file_2_view_0']
    col_map = {'transportation_cost_to_C1': 'C1', 'transportation_cost_to_C2': 'C2', 'transportation_cost_to_C3': 'C3', 'transportation_cost_to_C4': 'C4'}
    cost_rows = []
    for (source_row, row) in frame_cost.iterrows():
        supplier_id = row['supplier_id']
        if supplier_id not in I:
            raise ValueError(f'Supplier {supplier_id} in cost matrix not found in supply set')
        for (col, customer_id) in col_map.items():
            if customer_id not in J:
                raise ValueError(f'Customer {customer_id} in cost matrix not found in demand set')
            try:
                cost_val = float(row[col])
            except Exception:
                raise ValueError(f'Invalid cost value for ({supplier_id},{customer_id}): {row[col]}')
            c_ij[supplier_id, customer_id] = cost_val
            cost_rows.append((supplier_id, customer_id))
    for i in I:
        for j in J:
            if (i, j) not in c_ij:
                raise ValueError(f'Missing cost for ({i},{j})')
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(cost_rows, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((c_ij[i, j] * x_vars[i, j] for (i, j) in cost_rows)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) >= d_j[j])
    for i in I:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in J)) <= s_i[i])
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()