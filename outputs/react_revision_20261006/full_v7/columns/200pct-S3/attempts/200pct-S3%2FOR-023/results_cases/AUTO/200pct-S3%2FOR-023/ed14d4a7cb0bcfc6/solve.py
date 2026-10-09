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
 'tables': [{'columns': ['demand_previous_period',
                         'Customer',
                         'four_periods_ago_demand',
                         'three_periods_ago_demand',
                         'demand',
                         'two_periods_ago_demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1',
                                     'demand': '2397',
                                     'demand_previous_period': '2025',
                                     'four_periods_ago_demand': '2316',
                                     'three_periods_ago_demand': '2781',
                                     'two_periods_ago_demand': '2634'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2',
                                     'demand': '1889',
                                     'demand_previous_period': '1729',
                                     'four_periods_ago_demand': '2106',
                                     'three_periods_ago_demand': '1886',
                                     'two_periods_ago_demand': '1966'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3',
                                     'demand': '2518',
                                     'demand_previous_period': '2280',
                                     'four_periods_ago_demand': '2128',
                                     'three_periods_ago_demand': '2138',
                                     'two_periods_ago_demand': '2219'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4',
                                     'demand': '3218',
                                     'demand_previous_period': '3008',
                                     'four_periods_ago_demand': '3380',
                                     'three_periods_ago_demand': '3161',
                                     'two_periods_ago_demand': '3823'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5',
                                     'demand': '1813',
                                     'demand_previous_period': '1814',
                                     'four_periods_ago_demand': '1931',
                                     'three_periods_ago_demand': '2059',
                                     'two_periods_ago_demand': '1759'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['four_periods_ago_fixed_costs',
                         'two_periods_ago_fixed_costs',
                         'fixed_opening_cost_previous_period',
                         'Unnamed: 3',
                         'three_periods_ago_fixed_costs',
                         'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 3': 'MOUNT AYR',
                                     'fixed_costs': '96.58',
                                     'fixed_opening_cost_previous_period': '101.457290',
                                     'four_periods_ago_fixed_costs': '95.633516',
                                     'three_periods_ago_fixed_costs': '109.637616',
                                     'two_periods_ago_fixed_costs': '89.809742'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 3': 'WAUKEE',
                                     'fixed_costs': '94.06',
                                     'fixed_opening_cost_previous_period': '112.034866',
                                     'four_periods_ago_fixed_costs': '111.846746',
                                     'three_periods_ago_fixed_costs': '97.211010',
                                     'two_periods_ago_fixed_costs': '108.319496'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 3': 'WAVERLY',
                                     'fixed_costs': '94.37',
                                     'fixed_opening_cost_previous_period': '86.06544',
                                     'four_periods_ago_fixed_costs': '87.509301',
                                     'three_periods_ago_fixed_costs': '112.904268',
                                     'two_periods_ago_fixed_costs': '99.201744'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 3': 'PELLA',
                                     'fixed_costs': '82.88',
                                     'fixed_opening_cost_previous_period': '89.526976',
                                     'four_periods_ago_fixed_costs': '78.288448',
                                     'three_periods_ago_fixed_costs': '92.527232',
                                     'two_periods_ago_fixed_costs': '81.620224'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 3': 'DES MOINES',
                                     'fixed_costs': '94.95999999999999',
                                     'fixed_opening_cost_previous_period': '110.894288000',
                                     'four_periods_ago_fixed_costs': '77.4873600000',
                                     'three_periods_ago_fixed_costs': '109.109040000',
                                     'two_periods_ago_fixed_costs': '85.9388000000'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['two_periods_ago_CLARINDA',
                         'two_periods_ago_service_status',
                         'previous_period_SIOUX_CITY',
                         'previous_period_BANCROFT',
                         'Unnamed: 4',
                         'CLARINDA',
                         'two_periods_ago_FORT_MADISON',
                         'previous_period_FORT_MADISON',
                         'three_periods_ago_service_status',
                         'previous_period_service_status',
                         'previous_period_TOLEDO',
                         'FORT MADISON',
                         'previous_period_CLARINDA',
                         'SIOUX CITY',
                         'two_periods_ago_SIOUX_CITY',
                         'TOLEDO',
                         'four_periods_ago_service_status',
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
                                     'Unnamed: 4': 'MOUNT AYR',
                                     'four_periods_ago_service_status': 'Suspended',
                                     'previous_period_BANCROFT': '1741.995255',
                                     'previous_period_CLARINDA': '832.712916000',
                                     'previous_period_FORT_MADISON': '19.009500',
                                     'previous_period_SIOUX_CITY': '16.702254',
                                     'previous_period_TOLEDO': '197.766174',
                                     'previous_period_service_status': 'Seasonal',
                                     'three_periods_ago_service_status': 'Regular',
                                     'two_periods_ago_CLARINDA': '643.898892000',
                                     'two_periods_ago_FORT_MADISON': '15.929524',
                                     'two_periods_ago_SIOUX_CITY': '21.165822',
                                     'two_periods_ago_service_status': 'Seasonal'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 4': 'WAUKEE',
                                     'four_periods_ago_service_status': 'Suspended',
                                     'previous_period_BANCROFT': '92.494731',
                                     'previous_period_CLARINDA': '13.736527',
                                     'previous_period_FORT_MADISON': '1.75905',
                                     'previous_period_SIOUX_CITY': '1.480193',
                                     'previous_period_TOLEDO': '33.124228',
                                     'previous_period_service_status': 'Trial',
                                     'three_periods_ago_service_status': 'Trial',
                                     'two_periods_ago_CLARINDA': '14.809244',
                                     'two_periods_ago_FORT_MADISON': '1.69755',
                                     'two_periods_ago_SIOUX_CITY': '1.553981',
                                     'two_periods_ago_service_status': 'Trial'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 4': 'WAVERLY',
                                     'four_periods_ago_service_status': 'Regular',
                                     'previous_period_BANCROFT': '87.130491',
                                     'previous_period_CLARINDA': '1.975662',
                                     'previous_period_FORT_MADISON': '342.038794',
                                     'previous_period_SIOUX_CITY': '221.98932',
                                     'previous_period_TOLEDO': '44.98809',
                                     'previous_period_service_status': 'Regular',
                                     'three_periods_ago_service_status': 'Seasonal',
                                     'two_periods_ago_CLARINDA': '2.101788',
                                     'two_periods_ago_FORT_MADISON': '351.226436',
                                     'two_periods_ago_SIOUX_CITY': '223.34562',
                                     'two_periods_ago_service_status': 'Suspended'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 4': 'PELLA',
                                     'four_periods_ago_service_status': 'Seasonal',
                                     'previous_period_BANCROFT': '41.211298',
                                     'previous_period_CLARINDA': '960.05000',
                                     'previous_period_FORT_MADISON': '1520.663378',
                                     'previous_period_SIOUX_CITY': '1676.159116',
                                     'previous_period_TOLEDO': '1693.411545',
                                     'previous_period_service_status': 'Regular',
                                     'three_periods_ago_service_status': 'Trial',
                                     'two_periods_ago_CLARINDA': '1351.98672',
                                     'two_periods_ago_FORT_MADISON': '1560.481247',
                                     'two_periods_ago_SIOUX_CITY': '1695.586164',
                                     'two_periods_ago_service_status': 'Seasonal'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 4': 'DES MOINES',
                                     'four_periods_ago_service_status': 'Trial',
                                     'previous_period_BANCROFT': '109.094304',
                                     'previous_period_CLARINDA': '1093.98804',
                                     'previous_period_FORT_MADISON': '43.727836',
                                     'previous_period_SIOUX_CITY': '944.831319000',
                                     'previous_period_TOLEDO': '53.213173',
                                     'previous_period_service_status': 'Regular',
                                     'three_periods_ago_service_status': 'Seasonal',
                                     'two_periods_ago_CLARINDA': '877.10772',
                                     'two_periods_ago_FORT_MADISON': '48.701948',
                                     'two_periods_ago_SIOUX_CITY': '862.311264000',
                                     'two_periods_ago_service_status': 'Trial'}}],
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
    demand_df = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_df = CSVQA_FRAMES['file_1_view_0']
    trans_cost_df = CSVQA_FRAMES['file_2_view_0']
    suppliers = [row['Unnamed: 3'] for row in fixed_cost_df.to_dict(orient='records')]
    stores = [row['Customer'] for row in demand_df.to_dict(orient='records')]
    demand = {}
    for row in demand_df.to_dict(orient='records'):
        store = row['Customer']
        try:
            demand[store] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for store {store}: {row['demand']}")
    fixed_cost = {}
    for row in fixed_cost_df.to_dict(orient='records'):
        supplier = row['Unnamed: 3']
        try:
            fixed_cost[supplier] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier {supplier}: {row['fixed_costs']}")
    cost = {supplier: {} for supplier in suppliers}
    for row in trans_cost_df.to_dict(orient='records'):
        supplier = row['Unnamed: 4']
        for store in stores:
            store_col_map = ['BANCROFT', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO']
            store_idx = stores.index(store)
            if store_idx >= len(store_col_map):
                raise ValueError(f'Store index {store_idx} out of bounds for mapping.')
            col = store_col_map[store_idx]
            try:
                cost[supplier][store] = float(row[col])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {supplier}, store {store}: {row[col]}')
    M = sum((demand[store] for store in stores))
    if set(cost.keys()) != set(suppliers):
        raise ValueError('Mismatch in supplier keys between cost and suppliers.')
    for supplier in suppliers:
        if set(cost[supplier].keys()) != set(stores):
            raise ValueError(f'Mismatch in store keys for supplier {supplier} in cost matrix.')
    m = gp.Model('Iowa_Liquor_FLP')
    quantity_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * open_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in stores)) <= M * open_vars[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)