CSVQA_DATA = {'query': 'A supermarket manager needs to select a variety of products to stock in different sections of the store. Particularly, the store has several sections, each with a display space limit provided in "capacity.csv." The predefined price and shelf space requirement of each product are detailed in "products.csv." The objective is to determine the optimal number of units of each product to stock in each section to maximize the total revenue, while ensuring that the total space used by the products in each section does not exceed the available capacity. The decision variables x_ij denote the number of units of product j to be placed in section i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['SectionID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'SectionID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '150', 'SectionID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'SectionID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '130', 'SectionID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '90', 'SectionID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '110', 'SectionID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '160', 'SectionID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '140', 'SectionID': '8'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}}, {'data_row_number': 1, 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}}, {'data_row_number': 2, 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}}, {'data_row_number': 3, 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}}, {'data_row_number': 4, 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}}, {'data_row_number': 5, 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}}, {'data_row_number': 6, 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}}, {'data_row_number': 7, 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}}, {'data_row_number': 8, 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}}, {'data_row_number': 9, 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacities = {}
    values = {}
    weights = {}
    for table in tables:
        table_id = table.get('table_id')
        if table_id == 'file_0_view_0':
            for record in table['records']:
                vals = record['values']
                section_id = vals['SectionID']
                cap_str = vals['Capacity']
                if section_id in capacities:
                    raise ValueError(f'Duplicate SectionID {section_id}')
                try:
                    capacities[section_id] = int(cap_str)
                except Exception as e:
                    raise ValueError(f'Invalid capacity for SectionID {section_id}: {cap_str}') from e
        elif table_id == 'file_1_view_0':
            for record in table['records']:
                vals = record['values']
                prod_id = vals['ProductName']
                val_str = vals['Value']
                weight_str = vals['Weight']
                if prod_id in values or prod_id in weights:
                    raise ValueError(f'Duplicate ProductName {prod_id}')
                try:
                    values[prod_id] = int(val_str)
                    weights[prod_id] = int(weight_str)
                except Exception as e:
                    raise ValueError(f'Invalid value or weight for ProductName {prod_id}') from e
    S = set(capacities.keys())
    P = set(values.keys())
    for i in S:
        if i not in capacities:
            raise ValueError(f'Missing capacity for section {i}')
    for j in P:
        if j not in values or j not in weights:
            raise ValueError(f'Missing value or weight for product {j}')
    model = gurobipy.Model('section_product_allocation')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in S for j in P], vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((values[j] * x[i, j] for i in S for j in P)), GRB.MAXIMIZE)
    for i in S:
        model.addConstr(gurobipy.quicksum((weights[j] * x[i, j] for j in P)) <= capacities[i], name=f'cap_{i}')
    model.optimize()
    return model
m = solve_problem()
