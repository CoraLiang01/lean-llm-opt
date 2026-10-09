CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“FreshMart,” a supermarket chain, operates several warehouses that distribute fresh produce to its various '
          'retail locations. The daily demand for each store is outlined in “customer_demand.csv,” while the available '
          'supply capacity of each warehouse is provided in “supply_capacity.csv.” The transportation cost per unit of '
          'produce from each warehouse to each store is recorded in “transportation_costs.csv.” The goal is to '
          'determine the optimal amount of fresh produce to be shipped from each warehouse to each store, ensuring '
          'that all store demands are met without exceeding the warehouse capacities, while minimizing the total '
          'transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'Customers', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Suppliers', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['Customers', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Customers': 'Customer1', 'demand': '70'}},
                         {'source_row': 1, 'values': {'Customers': 'Customer2', 'demand': '80'}},
                         {'source_row': 2, 'values': {'Customers': 'Customer3', 'demand': '60'}},
                         {'source_row': 3, 'values': {'Customers': 'Customer4', 'demand': '90'}},
                         {'source_row': 4, 'values': {'Customers': 'Customer5', 'demand': '85'}},
                         {'source_row': 5, 'values': {'Customers': 'Customer6', 'demand': '95'}}],
             'returned_rows': 6,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Suppliers', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Suppliers': 'Supplier1', 'supply_capacity': '200'}},
                         {'source_row': 1, 'values': {'Suppliers': 'Supplier2', 'supply_capacity': '250'}},
                         {'source_row': 2, 'values': {'Suppliers': 'Supplier3', 'supply_capacity': '230'}},
                         {'source_row': 3, 'values': {'Suppliers': 'Supplier4', 'supply_capacity': '220'}},
                         {'source_row': 4, 'values': {'Suppliers': 'Supplier5', 'supply_capacity': '210'}}],
             'returned_rows': 5,
             'role': 'warehouse supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer1': '2',
                                     'Customer2': '3',
                                     'Customer3': '1',
                                     'Customer4': '2',
                                     'Customer5': '3',
                                     'Customer6': '2',
                                     'Unnamed: 0': 'Supplier1'}},
                         {'source_row': 1,
                          'values': {'Customer1': '1',
                                     'Customer2': '2',
                                     'Customer3': '3',
                                     'Customer4': '2',
                                     'Customer5': '3',
                                     'Customer6': '2',
                                     'Unnamed: 0': 'Supplier2'}},
                         {'source_row': 2,
                          'values': {'Customer1': '3',
                                     'Customer2': '1',
                                     'Customer3': '2',
                                     'Customer4': '3',
                                     'Customer5': '2',
                                     'Customer6': '3',
                                     'Unnamed: 0': 'Supplier3'}},
                         {'source_row': 3,
                          'values': {'Customer1': '2',
                                     'Customer2': '3',
                                     'Customer3': '2',
                                     'Customer4': '1',
                                     'Customer5': '3',
                                     'Customer6': '4',
                                     'Unnamed: 0': 'Supplier4'}},
                         {'source_row': 4,
                          'values': {'Customer1': '3',
                                     'Customer2': '2',
                                     'Customer3': '3',
                                     'Customer4': '3',
                                     'Customer5': '2',
                                     'Customer6': '3',
                                     'Unnamed: 0': 'Supplier5'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [5, 6],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 6]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    import pandas as pd
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    customers = []
    demand = {}
    for (_, row) in demand_frame.iterrows():
        cust = row['Customers']
        customers.append(cust)
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    suppliers = []
    supply_capacity = {}
    for (_, row) in supply_frame.iterrows():
        sup = row['Suppliers']
        suppliers.append(sup)
        try:
            supply_capacity[sup] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    cost_rows = set(cost_frame['Unnamed: 0'])
    cost_cols = [col for col in cost_frame.columns if col != 'Unnamed: 0']
    missing_rows = set(suppliers) - cost_rows
    missing_cols = set(customers) - set(cost_cols)
    if missing_rows:
        raise ValueError(f'Missing cost rows for suppliers: {missing_rows}')
    if missing_cols:
        raise ValueError(f'Missing cost columns for customers: {missing_cols}')
    for (_, row) in cost_frame.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            try:
                cost[sup][cust] = float(row[cust])
            except Exception:
                raise ValueError(f'Invalid cost value for supplier {sup}, customer {cust}: {row[cust]}')
    if set(demand.keys()) != set(customers):
        raise ValueError('Mismatch in customer demand keys and customer list')
    if set(supply_capacity.keys()) != set(suppliers):
        raise ValueError('Mismatch in supplier supply_capacity keys and supplier list')
    for sup in suppliers:
        if set(cost[sup].keys()) != set(customers):
            raise ValueError(f'Cost matrix missing customers for supplier {sup}')
    m = gp.Model('FreshMart_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()