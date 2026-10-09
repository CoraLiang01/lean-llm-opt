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
             'role': 'customer demand',
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

def solve_problem(CSVQA_FRAMES):
    df_demand = CSVQA_FRAMES['file_0_view_0']
    df_supply = CSVQA_FRAMES['file_1_view_0']
    df_cost = CSVQA_FRAMES['file_2_view_0']
    S = list(df_supply['Suppliers'])
    C = list(df_demand['Customers'])
    demand_c = {}
    for (_, row) in df_demand.iterrows():
        c = row['Customers']
        if c not in C:
            raise ValueError(f'Customer {c} in demand not in customer set')
        try:
            demand_c[c] = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {c}: {row['demand']}")
    supply_capacity_s = {}
    for (_, row) in df_supply.iterrows():
        s = row['Suppliers']
        if s not in S:
            raise ValueError(f'Supplier {s} in supply not in supplier set')
        try:
            supply_capacity_s[s] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Non-numeric supply_capacity for supplier {s}: {row['supply_capacity']}")
    cost_sc = {}
    for (_, row) in df_cost.iterrows():
        s = row['Unnamed: 0']
        if s not in S:
            raise ValueError(f'Supplier {s} in cost matrix not in supplier set')
        for c in C:
            if c not in df_cost.columns:
                raise ValueError(f'Customer {c} not found in cost matrix columns')
            try:
                cost_sc[s, c] = float(row[c])
            except Exception:
                raise ValueError(f'Non-numeric cost for ({s},{c}): {row[c]}')
    for s in S:
        for c in C:
            if (s, c) not in cost_sc:
                raise ValueError(f'Missing cost for ({s},{c})')
    for c in C:
        if c not in demand_c:
            raise ValueError(f'Missing demand for customer {c}')
    for s in S:
        if s not in supply_capacity_s:
            raise ValueError(f'Missing supply_capacity for supplier {s}')
    m = gp.Model('FreshMart_Transportation')
    quantity_vars = m.addVars([(s, c) for s in S for c in C], lb=0, vtype=GRB.CONTINUOUS, obj=[cost_sc[s, c] for s in S for c in C], name='')
    for c in C:
        m.addConstr(gp.quicksum((quantity_vars[s, c] for s in S)) == demand_c[c], name='')
    for s in S:
        m.addConstr(gp.quicksum((quantity_vars[s, c] for c in C)) <= supply_capacity_s[s], name='')
    m.ModelSense = GRB.MINIMIZE
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for (s, c) in quantity_vars.keys():
            print(f'x_{s}_{c}: {quantity_vars[s, c].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)