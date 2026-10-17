CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    I = []
    r_i = {}
    d_i = {}
    v_i = {}
    for rec in records:
        source_row = rec.get('source_row')
        if source_row is None:
            raise ValueError("Record missing 'source_row'")
        values = rec.get('values', {})
        try:
            revenue_str = values['Revenue']
            demand_str = values['Demand']
            inventory_str = values['Initial Inventory']
        except KeyError as e:
            raise ValueError(f'Missing required field {e} in record {source_row}')
        try:
            revenue = float(revenue_str)
        except ValueError:
            raise ValueError(f"Invalid Revenue value '{revenue_str}' in record {source_row}")
        try:
            demand = int(demand_str)
        except ValueError:
            raise ValueError(f"Invalid Demand value '{demand_str}' in record {source_row}")
        try:
            inventory = int(inventory_str)
        except ValueError:
            raise ValueError(f"Invalid Initial Inventory value '{inventory_str}' in record {source_row}")
        I.append(source_row)
        r_i[source_row] = revenue
        d_i[source_row] = demand
        v_i[source_row] = inventory
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x_vars = model.addVars(I, vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name='')
    for i in I:
        model.addConstr(x_vars[i] <= d_i[i])
        model.addConstr(x_vars[i] <= v_i[i])
    model.setObjective(gurobipy.quicksum((r_i[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()