CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail chain, ‚ÄúGreenMart,‚Äù operates several warehouses that supply products to its various store '
          'locations. The daily demand for each store is provided in ‚Äúcustomer_demand.csv,‚Äù while the daily supply '
          'capacity of each warehouse is detailed in ‚Äúsupply_capacity.csv.‚Äù The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in ‚Äútransportation_costs.csv.‚Äù The objective is '
          'to determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, '
          'ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_D1': 'D1',
                                          'transportation_cost_to_D2': 'D2',
                                          'transportation_cost_to_D3': 'D3',
                                          'transportation_cost_to_D4': 'D4',
                                          'transportation_cost_to_D5': 'D5'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'supplier_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'supplier_id',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'demand_units'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer_id': 'D1', 'demand_units': '428'}},
                         {'source_row': 1, 'values': {'customer_id': 'D2', 'demand_units': '217'}},
                         {'source_row': 2, 'values': {'customer_id': 'D3', 'demand_units': '214'}},
                         {'source_row': 3, 'values': {'customer_id': 'D4', 'demand_units': '380'}},
                         {'source_row': 4, 'values': {'customer_id': 'D5', 'demand_units': '254'}}],
             'returned_rows': 5,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'supply_capacity_units'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'supplier_id': 'S1', 'supply_capacity_units': '428'}},
                         {'source_row': 1, 'values': {'supplier_id': 'S2', 'supply_capacity_units': '217'}},
                         {'source_row': 2, 'values': {'supplier_id': 'S3', 'supply_capacity_units': '214'}},
                         {'source_row': 3, 'values': {'supplier_id': 'S4', 'supply_capacity_units': '380'}},
                         {'source_row': 4, 'values': {'supplier_id': 'S5', 'supply_capacity_units': '254'}}],
             'returned_rows': 5,
             'role': 'warehouse supply',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_D1',
                         'transportation_cost_to_D2',
                         'transportation_cost_to_D3',
                         'transportation_cost_to_D4',
                         'transportation_cost_to_D5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'supplier_id': 'S1',
                                     'transportation_cost_to_D1': '269.3910588020795',
                                     'transportation_cost_to_D2': '1.453733539093394',
                                     'transportation_cost_to_D3': '99.60345345756603',
                                     'transportation_cost_to_D4': '26.64078166309837',
                                     'transportation_cost_to_D5': '9.537688956880922'}},
                         {'source_row': 1,
                          'values': {'supplier_id': 'S2',
                                     'transportation_cost_to_D1': '9.291846876785185',
                                     'transportation_cost_to_D2': '10.874778437070225',
                                     'transportation_cost_to_D3': '144.52609291614627',
                                     'transportation_cost_to_D4': '11.420133077898234',
                                     'transportation_cost_to_D5': '153.1756819927813'}},
                         {'source_row': 2,
                          'values': {'supplier_id': 'S3',
                                     'transportation_cost_to_D1': '9.674584301671008',
                                     'transportation_cost_to_D2': '2.6191650959687944',
                                     'transportation_cost_to_D3': '100.8242249168735',
                                     'transportation_cost_to_D4': '3.212191088791688',
                                     'transportation_cost_to_D5': '133.8493396124168'}},
                         {'source_row': 3,
                          'values': {'supplier_id': 'S4',
                                     'transportation_cost_to_D1': '270.57498480010247',
                                     'transportation_cost_to_D2': '32.50253586',
                                     'transportation_cost_to_D3': '4.6842098096469815',
                                     'transportation_cost_to_D4': '1.5682269686546804',
                                     'transportation_cost_to_D5': '9.58927599'}},
                         {'source_row': 4,
                          'values': {'supplier_id': 'S5',
                                     'transportation_cost_to_D1': '226.0331910675782',
                                     'transportation_cost_to_D2': '8.669161980826471',
                                     'transportation_cost_to_D3': '65.47681316968448',
                                     'transportation_cost_to_D4': '9.068765258459958',
                                     'transportation_cost_to_D5': '202.65015316425533'}}],
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

def solve_problem(CSVQA_FRAMES):
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    warehouses = []
    supply_capacity = {}
    for (_, row) in supply_frame.iterrows():
        wid = row['supplier_id']
        warehouses.append(wid)
        try:
            supply_capacity[wid] = float(row['supply_capacity_units'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity_units for warehouse {wid}: {row['supply_capacity_units']}")
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    stores = []
    demand = {}
    for (_, row) in demand_frame.iterrows():
        sid = row['customer_id']
        stores.append(sid)
        try:
            demand[sid] = float(row['demand_units'])
        except Exception:
            raise ValueError(f"Invalid demand_units for store {sid}: {row['demand_units']}")
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost_columns = ['transportation_cost_to_D1', 'transportation_cost_to_D2', 'transportation_cost_to_D3', 'transportation_cost_to_D4', 'transportation_cost_to_D5']
    store_id_map = {'transportation_cost_to_D1': 'D1', 'transportation_cost_to_D2': 'D2', 'transportation_cost_to_D3': 'D3', 'transportation_cost_to_D4': 'D4', 'transportation_cost_to_D5': 'D5'}
    cost = {}
    for (_, row) in cost_frame.iterrows():
        wid = row['supplier_id']
        cost[wid] = {}
        for col in cost_columns:
            sid = store_id_map[col]
            try:
                cost[wid][sid] = float(row[col])
            except Exception:
                raise ValueError(f'Invalid cost for warehouse {wid} to store {sid}: {row[col]}')
    if set(warehouses) != set(cost.keys()):
        raise ValueError('Mismatch between warehouses in supply_capacity and transportation_costs.')
    for wid in warehouses:
        if set(stores) != set(cost[wid].keys()):
            raise ValueError(f'Mismatch between stores in demand and transportation_costs for warehouse {wid}.')
    m = gp.Model('GreenMart_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in warehouses for j in stores]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)