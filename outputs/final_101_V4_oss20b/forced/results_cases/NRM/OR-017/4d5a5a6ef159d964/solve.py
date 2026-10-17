CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'ZZ',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'ZZ'}],
                         'logic': 'and'},
             'original_rows': 5242,
             'records': [{'source_row': 5237,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}},
                         {'source_row': 5238,
                          'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}},
                         {'source_row': 5239,
                          'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}},
                         {'source_row': 5240,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}},
                         {'source_row': 5241,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == table_id:
            table = tbl
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    skus = []
    for record in table.get('records', []):
        values = record.get('values', {})
        sku = values.get('SKU')
        if sku is None:
            raise ValueError(f'Missing SKU in record {record.get('source_row')}')
        try:
            revenue = float(values.get('Revenue', '0'))
            demand = int(values.get('Demand', '0'))
            inventory = float(values.get('Initial Inventory', '0'))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid numeric value in record {record.get('source_row')}: {e}')
        skus.append(sku)
        revenue_dict[sku] = revenue
        demand_dict[sku] = demand
        inventory_dict[sku] = inventory
    for sku in skus:
        if sku not in revenue_dict or sku not in demand_dict or sku not in inventory_dict:
            raise ValueError(f'Missing parameter for SKU {sku}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(skus, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in skus)), GRB.MAXIMIZE)
    for i in skus:
        m.addConstr(x[i] <= demand_dict[i], name=f'demand_{i}')
    for i in skus:
        m.addConstr(x[i] <= inventory_dict[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()