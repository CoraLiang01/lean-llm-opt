CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The Iowa Department of Commerce requires that any store selling alcohol in bottled form for off-premises '
          'consumption must hold a Class “E” liquor license, a typical arrangement for most state liquor regulatory '
          'authorities. All alcohol sales from stores registered with the Iowa Department of Commerce are recorded in '
          'the department’s system, which is publicly released as open data by the State of Iowa. Several suppliers '
          'located in different cities can provide the necessary liquor products to these licensed stores. Each '
          'supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '“fixed_cost.csv.” The Department needs to source a unit of each liquor product for the stores from these '
          'suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded '
          'in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is '
          'provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for '
          'all liquor products across all licensed stores is met while minimizing the total cost. The decision '
          'variables y_i are binary, indicating whether a supplier is operational (open). The decision variables '
          'x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Customer': 'Customer_1', 'demand': '2397'}},
                         {'source_row': 1, 'values': {'Customer': 'Customer_2', 'demand': '1889'}},
                         {'source_row': 2, 'values': {'Customer': 'Customer_3', 'demand': '2518'}},
                         {'source_row': 3, 'values': {'Customer': 'Customer_4', 'demand': '3218'}},
                         {'source_row': 4, 'values': {'Customer': 'Customer_5', 'demand': '1813'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'MOUNT AYR', 'fixed_costs': '96.58'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'WAUKEE', 'fixed_costs': '94.06'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'WAVERLY', 'fixed_costs': '94.37'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'PELLA', 'fixed_costs': '82.88'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT'],
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
                                     'Unnamed: 0': 'MOUNT AYR'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 0': 'WAUKEE'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 0': 'WAVERLY'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 0': 'PELLA'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 0': 'DES MOINES'}}],
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
import pandas as pd

def solve_problem():
    frames = CSVQA_FRAMES
    suppliers = list(frames['file_1_view_0']['Unnamed: 0'])
    stores = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
    demand_df = frames['file_0_view_0']
    if len(demand_df) != len(stores):
        raise ValueError('Mismatch between number of stores in demand.csv and transportation_costs.csv columns.')
    d_j = {}
    for (idx, store) in enumerate(stores):
        demand_val = demand_df.iloc[idx]['demand']
        try:
            d_j[store] = float(demand_val)
        except Exception:
            raise ValueError(f'Invalid demand value for store {store}: {demand_val}')
    fixed_cost_df = frames['file_1_view_0']
    if list(fixed_cost_df['Unnamed: 0']) != suppliers:
        raise ValueError('Supplier order mismatch in fixed_cost.csv.')
    f_i = {}
    for (idx, supplier) in enumerate(suppliers):
        fixed_val = fixed_cost_df.iloc[idx]['fixed_costs']
        try:
            f_i[supplier] = float(fixed_val)
        except Exception:
            raise ValueError(f'Invalid fixed cost for supplier {supplier}: {fixed_val}')
    trans_df = frames['file_2_view_0']
    if list(trans_df['Unnamed: 0']) != suppliers:
        raise ValueError('Supplier order mismatch in transportation_costs.csv.')
    c_ij = {}
    for (i, supplier) in enumerate(suppliers):
        row = trans_df.iloc[i]
        for store in stores:
            cost_val = row[store]
            try:
                c_ij[supplier, store] = float(cost_val)
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {supplier}, store {store}: {cost_val}')
    for supplier in suppliers:
        if supplier not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {supplier}')
        for store in stores:
            if (supplier, store) not in c_ij:
                raise ValueError(f'Missing transportation cost for supplier {supplier}, store {store}')
    for store in stores:
        if store not in d_j:
            raise ValueError(f'Missing demand for store {store}')
    m = gp.Model('facility_location')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x_vars = m.addVars([(i, j) for i in suppliers for j in stores], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((f_i[i] * y_vars[i] for i in suppliers)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == d_j[j], name=f'demand_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()