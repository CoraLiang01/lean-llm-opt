CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The Iowa Department of Commerce requires that any store selling alcohol in bottled form for off-premises '
          'consumption must hold a Class ‚ÄúE‚Äù liquor license, a typical arrangement for most state liquor '
          'regulatory authorities. All alcohol sales from stores registered with the Iowa Department of Commerce are '
          'recorded in the department‚Äôs system, which is publicly released as open data by the State of Iowa. '
          'Several suppliers located in different cities can provide the necessary liquor products to these licensed '
          'stores. Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '‚Äúfixed_cost.csv.‚Äù The Department needs to source a unit of each liquor product for the stores from '
          'these suppliers. For each product, the transportation cost per unit from each supplier to each store is '
          'recorded in ‚Äútransportation_costs.csv.‚Äù Additionally, each store has a specific demand for these '
          'products, which is provided in ‚Äúdemand.csv.‚Äù The objective is to determine which suppliers to activate '
          'so that the demand for all liquor products across all licensed stores is met while minimizing the total '
          'cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The '
          'decision variables x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['demand_previous_period', 'Customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1', 'demand': '2397', 'demand_previous_period': '2025'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2', 'demand': '1889', 'demand_previous_period': '1729'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3', 'demand': '2518', 'demand_previous_period': '2280'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4', 'demand': '3218', 'demand_previous_period': '3008'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5', 'demand': '1813', 'demand_previous_period': '1814'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['fixed_opening_cost_previous_period', 'Unnamed: 1', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 1': 'MOUNT AYR',
                                     'fixed_costs': '96.58',
                                     'fixed_opening_cost_previous_period': '101.457290'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 1': 'WAUKEE',
                                     'fixed_costs': '94.06',
                                     'fixed_opening_cost_previous_period': '112.034866'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 1': 'WAVERLY',
                                     'fixed_costs': '94.37',
                                     'fixed_opening_cost_previous_period': '86.06544'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 1': 'PELLA',
                                     'fixed_costs': '82.88',
                                     'fixed_opening_cost_previous_period': '89.526976'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 1': 'DES MOINES',
                                     'fixed_costs': '94.95999999999999',
                                     'fixed_opening_cost_previous_period': '110.894288000'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0',
                         'CLARINDA',
                         'previous_period_FORT_MADISON',
                         'previous_period_service_status',
                         'FORT MADISON',
                         'previous_period_CLARINDA',
                         'SIOUX CITY',
                         'TOLEDO',
                         'BANCROFT'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'BANCROFT': '1685.53',
                                     'CLARINDA': '694.6799999999999',
                                     'FORT MADISON': '17.48',
                                     'SIOUX CITY': '20.07',
                                     'TOLEDO': '199.02',
                                     'Unnamed: 0': 'MOUNT AYR',
                                     'previous_period_CLARINDA': '832.712916000',
                                     'previous_period_FORT_MADISON': '19.009500',
                                     'previous_period_service_status': 'Seasonal'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 0': 'WAUKEE',
                                     'previous_period_CLARINDA': '13.736527',
                                     'previous_period_FORT_MADISON': '1.75905',
                                     'previous_period_service_status': 'Trial'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 0': 'WAVERLY',
                                     'previous_period_CLARINDA': '1.975662',
                                     'previous_period_FORT_MADISON': '342.038794',
                                     'previous_period_service_status': 'Regular'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 0': 'PELLA',
                                     'previous_period_CLARINDA': '960.05000',
                                     'previous_period_FORT_MADISON': '1520.663378',
                                     'previous_period_service_status': 'Regular'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 0': 'DES MOINES',
                                     'previous_period_CLARINDA': '1093.98804',
                                     'previous_period_FORT_MADISON': '43.727836',
                                     'previous_period_service_status': 'Regular'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 5], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 5], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    trans_cost_frame = CSVQA_FRAMES['file_2_view_0']
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    suppliers_fixed = [row['Unnamed: 1'] for (_, row) in fixed_cost_frame.iterrows()]
    suppliers_trans = [row['Unnamed: 0'] for (_, row) in trans_cost_frame.iterrows()]
    suppliers = []
    seen = set()
    for s in suppliers_fixed + suppliers_trans:
        if s not in seen:
            suppliers.append(s)
            seen.add(s)
    stores_demand = [row['Customer'] for (_, row) in demand_frame.iterrows()]
    store_cols_trans = [col for col in trans_cost_frame.columns if col not in ('Unnamed: 0', 'previous_period_CLARINDA', 'previous_period_FORT_MADISON', 'previous_period_service_status')]
    stores = []
    seen = set()
    for s in stores_demand + store_cols_trans:
        if s not in seen:
            stores.append(s)
            seen.add(s)
    demand_dict = {}
    for (_, row) in demand_frame.iterrows():
        store = row['Customer']
        demand_dict[store] = float(row['demand'])
    fixed_cost_dict = {}
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 1']
        fixed_cost_dict[supplier] = float(row['fixed_costs'])
    cost_dict = {}
    for (_, row) in trans_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        cost_dict[supplier] = {}
        for store in store_cols_trans:
            cost_dict[supplier][store] = float(row[store])
    D_dict = {}
    for store in stores:
        D_dict[store] = demand_dict.get(store, 0.0)
    for i in suppliers:
        if i not in fixed_cost_dict:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost_dict:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in stores:
            if j not in cost_dict[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    for j in stores:
        if j not in demand_dict:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('Iowa_FLP')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(i, j) for i in suppliers for j in stores]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost_dict[i][j] * x_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
    for i in suppliers:
        for j in stores:
            m.addConstr(x_vars[i, j] <= D_dict[j] * y_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')