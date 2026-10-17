CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    products = []
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    for rec in records:
        values = rec.get('values', {})
        product_name = values.get('Product Name')
        revenue_str = values.get('Revenue')
        demand_str = values.get('Demand')
        inventory_str = values.get('Initial Inventory')
        if product_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('source_row')}")
        if revenue_str is None:
            raise ValueError(f"Missing 'Revenue' in record {rec.get('source_row')}")
        if demand_str is None:
            raise ValueError(f"Missing 'Demand' in record {rec.get('source_row')}")
        if inventory_str is None:
            raise ValueError(f"Missing 'Initial Inventory' in record {rec.get('source_row')}")
        try:
            revenue = float(revenue_str)
        except ValueError:
            raise ValueError(f"Invalid revenue value '{revenue_str}' for product '{product_name}'")
        try:
            demand = float(demand_str)
        except ValueError:
            raise ValueError(f"Invalid demand value '{demand_str}' for product '{product_name}'")
        try:
            inventory = float(inventory_str)
        except ValueError:
            raise ValueError(f"Invalid inventory value '{inventory_str}' for product '{product_name}'")
        products.append(product_name)
        revenue_dict[product_name] = revenue
        demand_dict[product_name] = demand
        inventory_dict[product_name] = inventory
    for p in products:
        if p not in revenue_dict:
            raise ValueError(f'Revenue missing for product {p}')
        if p not in demand_dict:
            raise ValueError(f'Demand missing for product {p}')
        if p not in inventory_dict:
            raise ValueError(f'Inventory missing for product {p}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((revenue_dict[p] * x[p] for p in products)), GRB.MAXIMIZE)
    for p in products:
        model.addConstr(x[p] <= demand_dict[p], name=f'demand_{p}')
        model.addConstr(x[p] <= inventory_dict[p], name=f'inventory_{p}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()