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
 'tables': [{'columns': ['demand_previous_period', 'Customer', 'demand', 'two_periods_ago_demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1',
                                     'demand': '2397',
                                     'demand_previous_period': '2025',
                                     'two_periods_ago_demand': '2634'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2',
                                     'demand': '1889',
                                     'demand_previous_period': '1729',
                                     'two_periods_ago_demand': '1966'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3',
                                     'demand': '2518',
                                     'demand_previous_period': '2280',
                                     'two_periods_ago_demand': '2219'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4',
                                     'demand': '3218',
                                     'demand_previous_period': '3008',
                                     'two_periods_ago_demand': '3823'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5',
                                     'demand': '1813',
                                     'demand_previous_period': '1814',
                                     'two_periods_ago_demand': '1759'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['two_periods_ago_fixed_costs',
                         'fixed_opening_cost_previous_period',
                         'Unnamed: 2',
                         'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 2': 'MOUNT AYR',
                                     'fixed_costs': '96.58',
                                     'fixed_opening_cost_previous_period': '101.457290',
                                     'two_periods_ago_fixed_costs': '89.809742'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 2': 'WAUKEE',
                                     'fixed_costs': '94.06',
                                     'fixed_opening_cost_previous_period': '112.034866',
                                     'two_periods_ago_fixed_costs': '108.319496'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 2': 'WAVERLY',
                                     'fixed_costs': '94.37',
                                     'fixed_opening_cost_previous_period': '86.06544',
                                     'two_periods_ago_fixed_costs': '99.201744'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 2': 'PELLA',
                                     'fixed_costs': '82.88',
                                     'fixed_opening_cost_previous_period': '89.526976',
                                     'two_periods_ago_fixed_costs': '81.620224'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 2': 'DES MOINES',
                                     'fixed_costs': '94.95999999999999',
                                     'fixed_opening_cost_previous_period': '110.894288000',
                                     'two_periods_ago_fixed_costs': '85.9388000000'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['two_periods_ago_service_status',
                         'previous_period_SIOUX_CITY',
                         'Unnamed: 2',
                         'CLARINDA',
                         'previous_period_FORT_MADISON',
                         'previous_period_service_status',
                         'previous_period_TOLEDO',
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
                                     'Unnamed: 2': 'MOUNT AYR',
                                     'previous_period_CLARINDA': '832.712916000',
                                     'previous_period_FORT_MADISON': '19.009500',
                                     'previous_period_SIOUX_CITY': '16.702254',
                                     'previous_period_TOLEDO': '197.766174',
                                     'previous_period_service_status': 'Seasonal',
                                     'two_periods_ago_service_status': 'Seasonal'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 2': 'WAUKEE',
                                     'previous_period_CLARINDA': '13.736527',
                                     'previous_period_FORT_MADISON': '1.75905',
                                     'previous_period_SIOUX_CITY': '1.480193',
                                     'previous_period_TOLEDO': '33.124228',
                                     'previous_period_service_status': 'Trial',
                                     'two_periods_ago_service_status': 'Trial'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 2': 'WAVERLY',
                                     'previous_period_CLARINDA': '1.975662',
                                     'previous_period_FORT_MADISON': '342.038794',
                                     'previous_period_SIOUX_CITY': '221.98932',
                                     'previous_period_TOLEDO': '44.98809',
                                     'previous_period_service_status': 'Regular',
                                     'two_periods_ago_service_status': 'Suspended'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 2': 'PELLA',
                                     'previous_period_CLARINDA': '960.05000',
                                     'previous_period_FORT_MADISON': '1520.663378',
                                     'previous_period_SIOUX_CITY': '1676.159116',
                                     'previous_period_TOLEDO': '1693.411545',
                                     'previous_period_service_status': 'Regular',
                                     'two_periods_ago_service_status': 'Seasonal'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 2': 'DES MOINES',
                                     'previous_period_CLARINDA': '1093.98804',
                                     'previous_period_FORT_MADISON': '43.727836',
                                     'previous_period_SIOUX_CITY': '944.831319000',
                                     'previous_period_TOLEDO': '53.213173',
                                     'previous_period_service_status': 'Regular',
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
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    I = []
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 2']
        if supplier not in I:
            I.append(supplier)
    J = []
    for (_, row) in demand_frame.iterrows():
        customer = row['Customer']
        if customer not in J:
            J.append(customer)
    d_j = {}
    D_j = {}
    for (_, row) in demand_frame.iterrows():
        customer = row['Customer']
        demand = float(row['demand'])
        d_j[customer] = demand
        D_j[customer] = demand
    f_i = {}
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 2']
        fixed_cost = float(row['fixed_costs'])
        f_i[supplier] = fixed_cost
    c_ij = {}
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 2']
        c_ij[supplier] = {}
        for customer in J:
            cost_str = row.get(customer, '')
            if cost_str == '' or cost_str is None:
                raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {customer}')
            c_ij[supplier][customer] = float(cost_str)
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in c_ij:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('Iowa_FLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * x_vars[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((x_vars[i, j] <= D_j[j] * y_vars[i] for i in I for j in J), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')