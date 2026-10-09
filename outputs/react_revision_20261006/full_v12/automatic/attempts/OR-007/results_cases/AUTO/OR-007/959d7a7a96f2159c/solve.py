CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail chain, “GreenMart,” operates several warehouses that supply products to its various store '
          'locations. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply '
          'capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to '
          'determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, '
          'ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'region', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'D1', 'demand': '428'}},
                         {'source_row': 1, 'values': {'customer': 'D2', 'demand': '217'}},
                         {'source_row': 2, 'values': {'customer': 'D3', 'demand': '214'}},
                         {'source_row': 3, 'values': {'customer': 'D4', 'demand': '380'}},
                         {'source_row': 4, 'values': {'customer': 'D5', 'demand': '254'}}],
             'returned_rows': 5,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['region', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'region': 'S1', 'supply_capacity': '428'}},
                         {'source_row': 1, 'values': {'region': 'S2', 'supply_capacity': '217'}},
                         {'source_row': 2, 'values': {'region': 'S3', 'supply_capacity': '214'}},
                         {'source_row': 3, 'values': {'region': 'S4', 'supply_capacity': '380'}},
                         {'source_row': 4, 'values': {'region': 'S5', 'supply_capacity': '254'}}],
             'returned_rows': 5,
             'role': 'warehouse supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'D1': '269.3910588020795',
                                     'D2': '1.4537335390933939',
                                     'D3': '99.60345345756605',
                                     'D4': '26.64078166309837',
                                     'D5': '9.537688956880922',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'D1': '9.291846876785183',
                                     'D2': '10.874778437070223',
                                     'D3': '144.52609291614627',
                                     'D4': '11.420133077898234',
                                     'D5': '153.1756819927813',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'D1': '9.674584301671008',
                                     'D2': '2.6191650959687944',
                                     'D3': '100.8242249168735',
                                     'D4': '3.2121910887916876',
                                     'D5': '133.8493396124168',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'D1': '270.57498480010247',
                                     'D2': '32.50253586',
                                     'D3': '4.6842098096469815',
                                     'D4': '1.5682269686546804',
                                     'D5': '9.58927599',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'D1': '226.0331910675782',
                                     'D2': '8.669161980826471',
                                     'D3': '65.47681316968448',
                                     'D4': '9.068765258459958',
                                     'D5': '202.65015316425533',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
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
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    warehouses = []
    supply_capacity = {}
    for (_, row) in supply_frame.iterrows():
        warehouse = row['region']
        warehouses.append(warehouse)
        try:
            supply_capacity[warehouse] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity for warehouse {warehouse}: {row['supply_capacity']}")
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    stores = []
    demand = {}
    for (_, row) in demand_frame.iterrows():
        store = row['customer']
        stores.append(store)
        try:
            demand[store] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for store {store}: {row['demand']}")
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    matrix_warehouses = []
    matrix_stores = [col for col in cost_frame.columns if col != 'Unnamed: 0']
    for (_, row) in cost_frame.iterrows():
        warehouse = row['Unnamed: 0']
        matrix_warehouses.append(warehouse)
    if set(warehouses) != set(matrix_warehouses):
        raise ValueError('Mismatch between warehouses in supply_capacity.csv and transportation_costs.csv')
    if set(stores) != set(matrix_stores):
        raise ValueError('Mismatch between stores in customer_demand.csv and transportation_costs.csv')
    for (_, row) in cost_frame.iterrows():
        warehouse = row['Unnamed: 0']
        cost[warehouse] = {}
        for store in stores:
            try:
                cost[warehouse][store] = float(row[store])
            except Exception:
                raise ValueError(f'Invalid cost for warehouse {warehouse}, store {store}: {row[store]}')
    m = gp.Model('GreenMart_Transportation')
    quantity_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
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