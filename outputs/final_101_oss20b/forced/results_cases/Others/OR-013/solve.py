CSVQA_DATA = {'query': 'This supermarket offers a variety of best-selling products, and the specific revenue information is shown in the table, with relevant data provided in the “Revenue” column. Each product has its own demand level. The retailer’s objective is to maximize the total revenue by focusing on the sales volume of “4U” products. The initial inventory levels of these products are detailed in the “Initial Inventory” column. During the sales period, restocking is not allowed and there is no in-transit inventory.\n\nDemand for the “4U” products during the sales horizon is assumed to be deterministic and known in advance, with demand quantities specified in the “Demand” column. The decision variable x_i represents the number of units of each “4U” product i that the retailer plans to fulfill, where each x_i is a non-negative integer. Because the fulfillment quantity cannot exceed either the available inventory or the realized demand, the decision variables must satisfy both inventory and demand constraints.\n\nThe retailer therefore aims to determine the optimal fulfillment quantities in order to maximize total revenue while respecting both inventory availability and demand limits.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineSalesinUSA.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '5', 'Initial Inventory': '30', 'Product Name': '4U_Service_22', 'Revenue': '56.0'}}, {'data_row_number': 2, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': '4U_Service_36', 'Revenue': '21.6'}}, {'data_row_number': 3, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': '4U_Service_7', 'Revenue': '62.5'}}, {'data_row_number': 9421, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'COM4US5A9FCDA605F7B', 'Revenue': '10.0'}}, {'data_row_number': 9422, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'COM4US5A9FCF8FE2DE3', 'Revenue': '10.0'}}, {'data_row_number': 11027, 'values': {'Demand': '15', 'Initial Inventory': '100', 'Product Name': 'ENT4US5A979B09A2BF7', 'Revenue': '30.0'}}, {'data_row_number': 18013, 'values': {'Demand': '11', 'Initial Inventory': '70', 'Product Name': 'KAB4US59AD6CF40C781', 'Revenue': '12.0'}}, {'data_row_number': 18014, 'values': {'Demand': '36', 'Initial Inventory': '240', 'Product Name': 'KAB4US59AD6CF9E69D8', 'Revenue': '11.9'}}, {'data_row_number': 18015, 'values': {'Demand': '5', 'Initial Inventory': '30', 'Product Name': 'KAB4US59AD6D003A730', 'Revenue': '10.0'}}, {'data_row_number': 18016, 'values': {'Demand': '19', 'Initial Inventory': '130', 'Product Name': 'KAB4US59AD6D0489179', 'Revenue': '15.0'}}, {'data_row_number': 18017, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'KAB4US59AD6D0B32674', 'Revenue': '14.9'}}, {'data_row_number': 20681, 'values': {'Demand': '110', 'Initial Inventory': '710', 'Product Name': 'MAT4US5A8BEC8AB2742', 'Revenue': '45.0'}}, {'data_row_number': 20682, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'MAT4US5A9FAC1A50375', 'Revenue': '12.0'}}, {'data_row_number': 20683, 'values': {'Demand': '20', 'Initial Inventory': '130', 'Product Name': 'MAT4US5A9FB58986D2E', 'Revenue': '25.0'}}, {'data_row_number': 20684, 'values': {'Demand': '68', 'Initial Inventory': '480', 'Product Name': 'MAT4US5A9FBC6918099', 'Revenue': '5.0'}}, {'data_row_number': 20685, 'values': {'Demand': '9', 'Initial Inventory': '60', 'Product Name': 'MAT4US5A9FBDFFF21B5', 'Revenue': '7.0'}}, {'data_row_number': 20686, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'MAT4US5AABB7579D03C', 'Revenue': '80.0'}}, {'data_row_number': 37437, 'values': {'Demand': '19', 'Initial Inventory': '130', 'Product Name': 'WOF4US5A979D79EEF31', 'Revenue': '15.0'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    r = {}
    d = {}
    I = {}
    for rec in records:
        key = rec['data_row_number']
        vals = rec['values']
        for col in ('Revenue', 'Demand', 'Initial Inventory'):
            if col not in vals:
                raise ValueError(f"Missing column '{col}' in record {key}")
        try:
            revenue = float(vals['Revenue'])
            demand = int(vals['Demand'])
            inventory = int(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {key}: {e}')
        r[key] = revenue
        d[key] = demand
        I[key] = inventory
    model = gurobipy.Model('MaxRevenue4U')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(r.keys(), vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((r[i] * x[i] for i in r.keys())), GRB.MAXIMIZE)
    for i in r.keys():
        model.addConstr(x[i] <= d[i], name=f'demand_{i}')
        model.addConstr(x[i] <= I[i], name=f'inventory_{i}')
    model.optimize()
    return model
m = solve_problem()
