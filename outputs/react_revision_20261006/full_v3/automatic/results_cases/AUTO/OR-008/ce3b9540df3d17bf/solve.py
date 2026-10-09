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
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    suppliers_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
    customers_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
    suppliers = [rec['values']['Suppliers'] for rec in suppliers_table['records']]
    customers = [rec['values']['Customers'] for rec in customers_table['records']]
    demand = {}
    for rec in customers_table['records']:
        j = rec['values']['Customers']
        d = rec['values']['demand']
        try:
            demand[j] = float(d)
        except Exception:
            raise ValueError(f'Invalid demand value for customer {j}: {d}')
    supply_capacity = {}
    for rec in suppliers_table['records']:
        i = rec['values']['Suppliers']
        s = rec['values']['supply_capacity']
        try:
            supply_capacity[i] = float(s)
        except Exception:
            raise ValueError(f'Invalid supply capacity for supplier {i}: {s}')
    cost = {}
    row_label_to_supplier = {}
    for rec in suppliers_table['records']:
        row_label_to_supplier[rec['values']['Suppliers']] = rec['values']['Suppliers']
    for rec in cost_table['records']:
        i = rec['values']['Unnamed: 0']
        if i not in suppliers:
            raise ValueError(f'Supplier {i} in cost matrix not found in suppliers list')
        cost[i] = {}
        for j in customers:
            if j not in rec['values']:
                raise ValueError(f'Customer {j} not found in cost matrix columns')
            cij = rec['values'][j]
            try:
                cost[i][j] = float(cij)
            except Exception:
                raise ValueError(f'Invalid cost value for ({i},{j}): {cij}')
    if set(demand.keys()) != set(customers):
        raise ValueError('Mismatch in customer demand keys and customers list')
    if set(supply_capacity.keys()) != set(suppliers):
        raise ValueError('Mismatch in supply capacity keys and suppliers list')
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        if set(cost[i].keys()) != set(customers):
            raise ValueError(f'Cost row for supplier {i} missing customers')
    m = gp.Model('FreshMart_Transportation')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()