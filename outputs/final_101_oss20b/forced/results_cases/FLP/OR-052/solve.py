CSVQA_DATA = {'query': 'In the context of a bookstore, the store needs to allocate various types of books into different bookshelves. Specifically, the store has several bookshelves, each with a capacity limit provided in “capacity.csv.” The predefined value and weight of each book can be found in “products.csv.” The objective is to determine the optimal number of units of each book to place on each bookshelf to maximize the total value of the books across all bookshelves while ensuring that the total weight of the books on each bookshelf does not exceed its capacity. The decision variables x_ij represent the number of units of book j to be placed on bookshelf i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['BookshelfID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'BookshelfID': '1', 'Capacity': '200'}}, {'data_row_number': 1, 'values': {'BookshelfID': '2', 'Capacity': '200'}}, {'data_row_number': 2, 'values': {'BookshelfID': '3', 'Capacity': '300'}}, {'data_row_number': 3, 'values': {'BookshelfID': '4', 'Capacity': '400'}}, {'data_row_number': 4, 'values': {'BookshelfID': '5', 'Capacity': '550'}}, {'data_row_number': 5, 'values': {'BookshelfID': '6', 'Capacity': '600'}}, {'data_row_number': 6, 'values': {'BookshelfID': '7', 'Capacity': '650'}}, {'data_row_number': 7, 'values': {'BookshelfID': '8', 'Capacity': '750'}}, {'data_row_number': 8, 'values': {'BookshelfID': '9', 'Capacity': '820'}}, {'data_row_number': 9, 'values': {'BookshelfID': '10', 'Capacity': '570'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'The Great Gatsby', 'Value': '50', 'Weight': '10'}}, {'data_row_number': 1, 'values': {'ProductName': 'To Kill a Mockingbird', 'Value': '70', 'Weight': '20'}}, {'data_row_number': 2, 'values': {'ProductName': '1984', 'Value': '30', 'Weight': '5'}}, {'data_row_number': 3, 'values': {'ProductName': 'Pride and Prejudice', 'Value': '60', 'Weight': '15'}}, {'data_row_number': 4, 'values': {'ProductName': 'The Catcher in the Rye', 'Value': '80', 'Weight': '25'}}, {'data_row_number': 5, 'values': {'ProductName': 'Moby Dick', 'Value': '90', 'Weight': '30'}}, {'data_row_number': 6, 'values': {'ProductName': 'Jane Eyre', 'Value': '40', 'Weight': '12'}}, {'data_row_number': 7, 'values': {'ProductName': 'War and Peace', 'Value': '100', 'Weight': '35'}}, {'data_row_number': 8, 'values': {'ProductName': 'The Odyssey', 'Value': '55', 'Weight': '10'}}, {'data_row_number': 9, 'values': {'ProductName': 'Crime and Punishment', 'Value': '75', 'Weight': '20'}}, {'data_row_number': 10, 'values': {'ProductName': 'The Hobbit', 'Value': '65', 'Weight': '18'}}, {'data_row_number': 11, 'values': {'ProductName': 'Brave New World', 'Value': '95', 'Weight': '28'}}, {'data_row_number': 12, 'values': {'ProductName': 'Anna Karenina', 'Value': '45', 'Weight': '8'}}, {'data_row_number': 13, 'values': {'ProductName': 'Wuthering Heights', 'Value': '85', 'Weight': '22'}}, {'data_row_number': 14, 'values': {'ProductName': 'The Divine Comedy', 'Value': '70', 'Weight': '25'}}, {'data_row_number': 15, 'values': {'ProductName': 'The Iliad', 'Value': '110', 'Weight': '40'}}, {'data_row_number': 16, 'values': {'ProductName': 'Les Misérables', 'Value': '50', 'Weight': '14'}}, {'data_row_number': 17, 'values': {'ProductName': 'Dracula', 'Value': '60', 'Weight': '16'}}, {'data_row_number': 18, 'values': {'ProductName': 'Frankenstein', 'Value': '120', 'Weight': '50'}}, {'data_row_number': 19, 'values': {'ProductName': 'The Brothers Karamazov', 'Value': '100', 'Weight': '30'}}, {'data_row_number': 20, 'values': {'ProductName': 'Don Quixote', 'Value': '52', 'Weight': '11'}}, {'data_row_number': 21, 'values': {'ProductName': 'One Hundred Years of Solitude', 'Value': '68', 'Weight': '19'}}, {'data_row_number': 22, 'values': {'ProductName': 'Ulysses', 'Value': '38', 'Weight': '7'}}, {'data_row_number': 23, 'values': {'ProductName': 'The Alchemist', 'Value': '58', 'Weight': '14'}}, {'data_row_number': 24, 'values': {'ProductName': 'Meditations', 'Value': '82', 'Weight': '24'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    capacity = {}
    for rec in cap_table['records']:
        key = rec['values']['BookshelfID']
        val = rec['values']['Capacity']
        try:
            capacity[key] = int(val)
        except ValueError:
            raise ValueError(f'Invalid capacity value for bookshelf {key}')
    value = {}
    weight = {}
    for rec in prod_table['records']:
        key = rec['values']['ProductName']
        v = rec['values']['Value']
        w = rec['values']['Weight']
        try:
            value[key] = int(v)
            weight[key] = int(w)
        except ValueError:
            raise ValueError(f'Invalid value/weight for product {key}')
    I = list(capacity.keys())
    J = list(value.keys())
    for i in I:
        if i not in capacity:
            raise ValueError(f'Missing capacity for bookshelf {i}')
    for j in J:
        if j not in value or j not in weight:
            raise ValueError(f'Missing value or weight for product {j}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((value[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(gurobipy.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i], name=f'cap_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
