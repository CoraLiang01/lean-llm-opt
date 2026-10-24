CSVQA_DATA = {'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RestaurantSalesreport.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1483', 'Initial Inventory': '10440.0', 'Product Name': 'Aalopuri', 'Revenue': '20'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    products = []
    R = {}
    D = {}
    S = {}
    for rec in records:
        values = rec['values']
        prod_name = values['Product Name'].strip()
        if 'Revenue' not in values or 'Demand' not in values or 'Initial Inventory' not in values:
            raise ValueError(f"Missing parameter in record {rec['data_row_number']}")
        try:
            revenue = float(values['Revenue'])
            demand = float(values['Demand'])
            inventory = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec['data_row_number']}: {e}")
        products.append(prod_name)
        R[prod_name] = revenue
        D[prod_name] = demand
        S[prod_name] = inventory
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((R[i] * x[i] for i in products)), GRB.MAXIMIZE)
    for i in products:
        m.addConstr(x[i] <= D[i], name=f'dem_{i}')
        m.addConstr(x[i] <= S[i], name=f'inv_{i}')
    m.optimize()
    return m
m = solve_problem()
