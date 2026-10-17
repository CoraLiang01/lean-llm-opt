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
 'route': 'NRM',
 'tables': [{'columns': ['Customers', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Customers': 'Customer1', 'demand': '70'}},
                         {'source_row': 1, 'values': {'Customers': 'Customer2', 'demand': '80'}},
                         {'source_row': 2, 'values': {'Customers': 'Customer3', 'demand': '60'}},
                         {'source_row': 3, 'values': {'Customers': 'Customer4', 'demand': '90'}},
                         {'source_row': 4, 'values': {'Customers': 'Customer5', 'demand': '85'}},
                         {'source_row': 5, 'values': {'Customers': 'Customer6', 'demand': '95'}}],
             'returned_rows': 6,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Suppliers', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Suppliers': 'Supplier1', 'supply_capacity': '200'}},
                         {'source_row': 1, 'values': {'Suppliers': 'Supplier2', 'supply_capacity': '250'}},
                         {'source_row': 2, 'values': {'Suppliers': 'Supplier3', 'supply_capacity': '230'}},
                         {'source_row': 3, 'values': {'Suppliers': 'Supplier4', 'supply_capacity': '220'}},
                         {'source_row': 4, 'values': {'Suppliers': 'Supplier5', 'supply_capacity': '210'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [5, 6],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 6]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise KeyError(f'Table {table_id} not found')
    demand_table = get_table('file_0_view_0')
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['Customers']
        val = float(rec['values']['demand'])
        demand[cust] = val
    supply_table = get_table('file_1_view_0')
    supply = {}
    for rec in supply_table['records']:
        sup = rec['values']['Suppliers']
        val = float(rec['values']['supply_capacity'])
        supply[sup] = val
    cost_table = get_table('file_2_view_0')
    cost = {}
    cost_columns = [c for c in cost_table['columns'] if c != 'Unnamed: 0']
    for rec in cost_table['records']:
        sup = rec['values']['Unnamed: 0']
        for cust in cost_columns:
            val = float(rec['values'][cust])
            cost[sup, cust] = val
    if set(demand.keys()) != set(cost_columns):
        missing = set(cost_columns) - set(demand.keys())
        if missing:
            raise ValueError(f'Missing demand for customers: {missing}')
        missing = set(demand.keys()) - set(cost_columns)
        if missing:
            raise ValueError(f'Missing cost columns for customers: {missing}')
    if set(supply.keys()) != set(cost.keys(), key=lambda x: x[0]):
        cost_suppliers = {sup for sup, _ in cost.keys()}
        missing = cost_suppliers - set(supply.keys())
        if missing:
            raise ValueError(f'Missing supply for suppliers: {missing}')
        missing = set(supply.keys()) - cost_suppliers
        if missing:
            raise ValueError(f'Missing cost rows for suppliers: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    suppliers = list(supply.keys())
    customers = list(demand.keys())
    x = m.addVars([(i, j) for i in suppliers for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in customers)) <= supply[i] for i in suppliers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()