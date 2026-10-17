CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“FreshMart,” a supermarket chain, operates several warehouses that distribute fresh produce to its various '
          'retail locations. The daily demand for each store is outlined in “customer_demand.csv,” while the available '
          'supply capacity of each warehouse is provided in “supply_capacity.csv.” The transportation cost per unit of '
          'produce from each warehouse to each store is recorded in “transportation_costs.csv.” The goal is to '
          'determine the optimal amount of fresh produce to be shipped from each warehouse to each store, ensuring '
          'that all store demands are met without exceeding the warehouse capacities, while minimizing the total '
          'transportation cost.',
 'relationships': [],
 'route': 'RA',
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
             'role': 'file_0',
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
             'role': 'file_1',
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
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 4], "
                                   "'expected_shape': [5, 6], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 4], "
                                   "'expected_shape': [5, 6], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t.get('table_id') == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    demand_table = get_table('file_0_view_0')
    d = {}
    for rec in demand_table['records']:
        cust = rec['values']['Customers']
        val = float(rec['values']['demand'])
        d[cust] = val
    supply_table = get_table('file_1_view_0')
    s = {}
    for rec in supply_table['records']:
        sup = rec['values']['Suppliers']
        val = float(rec['values']['supply_capacity'])
        s[sup] = val
    cost_table = get_table('file_2_view_0')
    cost_cols = [col for col in cost_table['columns'] if col != 'Unnamed: 0']
    if set(cost_cols) != set(d.keys()):
        missing = set(d.keys()) - set(cost_cols)
        extra = set(cost_cols) - set(d.keys())
        msg = []
        if missing:
            msg.append(f'Missing cost columns for customers: {missing}')
        if extra:
            msg.append(f'Unexpected cost columns: {extra}')
        raise ValueError('; '.join(msg))
    c = {}
    for rec in cost_table['records']:
        sup = rec['values']['Unnamed: 0']
        if sup not in s:
            raise ValueError(f'Supplier {sup} in cost matrix not found in supply table')
        c[sup] = {}
        for cust in cost_cols:
            val = float(rec['values'][cust])
            c[sup][cust] = val
    for sup in s:
        if sup not in c:
            raise ValueError(f'Supplier {sup} missing in cost matrix')
        for cust in d:
            if cust not in c[sup]:
                raise ValueError(f'Cost for supplier {sup} to customer {cust} missing')
    m = gurobipy.Model('transportation')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(sup, cust) for sup in s for cust in d], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((c[sup][cust] * x[sup, cust] for sup in s for cust in d)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[sup, cust] for sup in s)) >= d[cust] for cust in d), name='')
    m.addConstrs((gurobipy.quicksum((x[sup, cust] for cust in d)) <= s[sup] for sup in s), name='')
    m.optimize()
    return m
m = solve_problem()