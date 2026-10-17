CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of a bookstore, the store needs to allocate various types of books into different '
          'bookshelves. Specifically, the store has several bookshelves, each with a capacity limit provided in '
          '“capacity.csv.” The predefined value and weight of each book can be found in “products.csv.” The objective '
          'is to determine the optimal number of units of each book to place on each bookshelf to maximize the total '
          'value of the books across all bookshelves while ensuring that the total weight of the books on each '
          'bookshelf does not exceed its capacity. The decision variables x_ij represent the number of units of book j '
          'to be placed on bookshelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['BookshelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'BookshelfID': '1', 'Capacity': '200'}},
                         {'source_row': 1, 'values': {'BookshelfID': '2', 'Capacity': '200'}},
                         {'source_row': 2, 'values': {'BookshelfID': '3', 'Capacity': '300'}},
                         {'source_row': 3, 'values': {'BookshelfID': '4', 'Capacity': '400'}},
                         {'source_row': 4, 'values': {'BookshelfID': '5', 'Capacity': '550'}},
                         {'source_row': 5, 'values': {'BookshelfID': '6', 'Capacity': '600'}},
                         {'source_row': 6, 'values': {'BookshelfID': '7', 'Capacity': '650'}},
                         {'source_row': 7, 'values': {'BookshelfID': '8', 'Capacity': '750'}},
                         {'source_row': 8, 'values': {'BookshelfID': '9', 'Capacity': '820'}},
                         {'source_row': 9, 'values': {'BookshelfID': '10', 'Capacity': '570'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 25,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'The Great Gatsby', 'Value': '50', 'Weight': '10'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'To Kill a Mockingbird', 'Value': '70', 'Weight': '20'}},
                         {'source_row': 2, 'values': {'ProductName': '1984', 'Value': '30', 'Weight': '5'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Pride and Prejudice', 'Value': '60', 'Weight': '15'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'The Catcher in the Rye', 'Value': '80', 'Weight': '25'}},
                         {'source_row': 5, 'values': {'ProductName': 'Moby Dick', 'Value': '90', 'Weight': '30'}},
                         {'source_row': 6, 'values': {'ProductName': 'Jane Eyre', 'Value': '40', 'Weight': '12'}},
                         {'source_row': 7, 'values': {'ProductName': 'War and Peace', 'Value': '100', 'Weight': '35'}},
                         {'source_row': 8, 'values': {'ProductName': 'The Odyssey', 'Value': '55', 'Weight': '10'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Crime and Punishment', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 10, 'values': {'ProductName': 'The Hobbit', 'Value': '65', 'Weight': '18'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Brave New World', 'Value': '95', 'Weight': '28'}},
                         {'source_row': 12, 'values': {'ProductName': 'Anna Karenina', 'Value': '45', 'Weight': '8'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Wuthering Heights', 'Value': '85', 'Weight': '22'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'The Divine Comedy', 'Value': '70', 'Weight': '25'}},
                         {'source_row': 15, 'values': {'ProductName': 'The Iliad', 'Value': '110', 'Weight': '40'}},
                         {'source_row': 16, 'values': {'ProductName': 'Les Misérables', 'Value': '50', 'Weight': '14'}},
                         {'source_row': 17, 'values': {'ProductName': 'Dracula', 'Value': '60', 'Weight': '16'}},
                         {'source_row': 18, 'values': {'ProductName': 'Frankenstein', 'Value': '120', 'Weight': '50'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'The Brothers Karamazov', 'Value': '100', 'Weight': '30'}},
                         {'source_row': 20, 'values': {'ProductName': 'Don Quixote', 'Value': '52', 'Weight': '11'}},
                         {'source_row': 21,
                          'values': {'ProductName': 'One Hundred Years of Solitude', 'Value': '68', 'Weight': '19'}},
                         {'source_row': 22, 'values': {'ProductName': 'Ulysses', 'Value': '38', 'Weight': '7'}},
                         {'source_row': 23, 'values': {'ProductName': 'The Alchemist', 'Value': '58', 'Weight': '14'}},
                         {'source_row': 24, 'values': {'ProductName': 'Meditations', 'Value': '82', 'Weight': '24'}}],
             'returned_rows': 25,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if cap_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA")
    if prod_table is None:
        raise ValueError("Product table 'file_1_view_0' not found in CSVQA_DATA")
    c_i = {}
    for rec in cap_table['records']:
        sid = rec['values']['BookshelfID']
        cap = int(rec['values']['Capacity'])
        if sid in c_i:
            raise ValueError(f"Duplicate BookshelfID '{sid}' in capacity data")
        c_i[sid] = cap
    if len(c_i) != len(cap_table['records']):
        raise ValueError('Mismatch in capacity records count')
    v_j = {}
    w_j = {}
    for rec in prod_table['records']:
        pname = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        if pname in v_j:
            raise ValueError(f"Duplicate ProductName '{pname}' in product data")
        v_j[pname] = val
        w_j[pname] = wt
    if len(v_j) != len(prod_table['records']):
        raise ValueError('Mismatch in product records count')
    shelves = list(c_i.keys())
    books = list(v_j.keys())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in shelves for j in books], lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((v_j[j] * x[i, j] for i in shelves for j in books)), GRB.MAXIMIZE)
    model.addConstrs((gurobipy.quicksum((w_j[j] * x[i, j] for j in books)) <= c_i[i] for i in shelves), name='')
    model.optimize()
    return model
m = solve_problem()