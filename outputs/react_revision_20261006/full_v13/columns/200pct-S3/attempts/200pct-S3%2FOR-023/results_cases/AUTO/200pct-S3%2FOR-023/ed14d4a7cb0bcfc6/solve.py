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
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 3', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 3': 'MOUNT AYR', 'fixed_costs': '96.58'}},
                         {'source_row': 1, 'values': {'Unnamed: 3': 'WAUKEE', 'fixed_costs': '94.06'}},
                         {'source_row': 2, 'values': {'Unnamed: 3': 'WAVERLY', 'fixed_costs': '94.37'}},
                         {'source_row': 3, 'values': {'Unnamed: 3': 'PELLA', 'fixed_costs': '82.88'}},
                         {'source_row': 4, 'values': {'Unnamed: 3': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 4', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT'],
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
                                     'Unnamed: 4': 'MOUNT AYR'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 4': 'WAUKEE'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 4': 'WAVERLY'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 4': 'PELLA'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 4': 'DES MOINES'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    trans_cost_frame = CSVQA_FRAMES['file_2_view_0']
    suppliers_fixed = set()
    for (_, row) in fixed_cost_frame.iterrows():
        suppliers_fixed.add(row['Unnamed: 3'])
    suppliers_trans = set()
    for (_, row) in trans_cost_frame.iterrows():
        suppliers_trans.add(row['Unnamed: 4'])
    suppliers = sorted(suppliers_fixed | suppliers_trans)
    stores_demand = set()
    for (_, row) in demand_frame.iterrows():
        stores_demand.add(row['Customer'])
    stores_trans = set(trans_cost_frame.columns) - {'Unnamed: 4'}
    stores = sorted(stores_demand | stores_trans)
    demand = {}
    for (_, row) in demand_frame.iterrows():
        j = row['Customer']
        try:
            demand[j] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for store {j}: {row['demand']}")
    for j in stores:
        if j not in demand:
            demand[j] = 0.0
    fixed_cost = {}
    for (_, row) in fixed_cost_frame.iterrows():
        i = row['Unnamed: 3']
        try:
            fixed_cost[i] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier {i}: {row['fixed_costs']}")
    for i in suppliers:
        if i not in fixed_cost:
            fixed_cost[i] = 0.0
    cost = {}
    for (_, row) in trans_cost_frame.iterrows():
        i = row['Unnamed: 4']
        cost[i] = {}
        for j in stores:
            if j in trans_cost_frame.columns:
                val = row[j]
                try:
                    cost[i][j] = float(val)
                except Exception:
                    raise ValueError(f'Invalid transportation cost for supplier {i}, store {j}: {val}')
            else:
                cost[i][j] = 1000000000.0
    M = sum((demand[j] for j in stores))
    M_i = {i: M for i in suppliers}
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    m = gp.Model('Iowa_Liquor_FLP')
    x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= M_i[i] * y_vars[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()