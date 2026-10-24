CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesdata.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '3066513', 'Initial Inventory': '22749210', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records returned for the filtered view')
    I = []
    r = {}
    d = {}
    v = {}
    for rec in records:
        vals = rec.get('values', {})
        prod_name = vals.get('Product Name')
        if prod_name is None:
            raise ValueError("Missing 'Product Name' in record")
        prod_name = prod_name.strip()
        if not prod_name.startswith('Baby'):
            raise ValueError(f"Record '{prod_name}' does not satisfy prefix filter")
        revenue_str = vals.get('Revenue')
        demand_str = vals.get('Demand')
        inv_str = vals.get('Initial Inventory')
        if revenue_str is None or demand_str is None or inv_str is None:
            raise ValueError(f"Missing required fields in record '{prod_name}'")
        try:
            revenue = float(revenue_str)
            demand = int(demand_str)
            inventory = int(inv_str)
        except ValueError as e:
            raise ValueError(f"Numeric conversion error in record '{prod_name}': {e}")
        I.append(prod_name)
        r[prod_name] = revenue
        d[prod_name] = demand
        v[prod_name] = inventory
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, name='')
    for i in I:
        model.addConstr(x[i] <= d[i], name=f'demand_{i}')
        model.addConstr(x[i] <= v[i], name=f'inventory_{i}')
    model.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
