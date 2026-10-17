CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the “fixed_cost.csv” file. Each branch needs to source a '
          'certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of goods '
          'from each supplier is recorded in the “transportation_costs.csv” file. Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all branches "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '143'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '6'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '10'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '25'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '3'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'C1': '150.74',
                                     'C2': '0.02',
                                     'C3': '49.13',
                                     'C4': '2080.15',
                                     'C5': '426.4',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '233.05',
                                     'C2': '97.73',
                                     'C3': '49.84',
                                     'C4': '1982.39',
                                     'C5': '23.96',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '55.68',
                                     'C2': '935.61',
                                     'C3': '4.03',
                                     'C4': '73.09',
                                     'C5': '525.3200000000001',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1483.82',
                                     'C2': '1801.08',
                                     'C3': '112.16',
                                     'C4': '816.05',
                                     'C5': '107.01',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1119.47',
                                     'C2': '884.3099999999999',
                                     'C3': '0.08',
                                     'C4': '1544.95',
                                     'C5': '543.67',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    supplier_table = tables['file_1_view_0']
    I = [rec['values']['Unnamed: 0'] for rec in supplier_table['records']]
    f = {}
    for rec in supplier_table['records']:
        key = rec['values']['Unnamed: 0']
        val = rec['values']['fixed_costs']
        try:
            f[key] = float(val)
        except ValueError:
            raise ValueError(f'Invalid fixed cost for supplier {key}')
    branch_table = tables['file_0_view_0']
    J = [rec['values']['customer'] for rec in branch_table['records']]
    d = {}
    for rec in branch_table['records']:
        key = rec['values']['customer']
        val = rec['values']['demand']
        try:
            d[key] = float(val)
        except ValueError:
            raise ValueError(f'Invalid demand for branch {key}')
    cost_table = tables['file_2_view_0']
    c = {}
    for rec in cost_table['records']:
        supplier = rec['values']['Unnamed: 0']
        if supplier not in I:
            raise ValueError(f'Supplier {supplier} in cost matrix not in supplier list')
        for branch in J:
            if branch not in rec['values']:
                raise ValueError(f'Branch {branch} missing in cost matrix row for supplier {supplier}')
            val = rec['values'][branch]
            try:
                c[supplier, branch] = float(val)
            except ValueError:
                raise ValueError(f'Invalid cost for supplier {supplier} to branch {branch}')
    for i in I:
        if i not in f:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for branch {j}')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for supplier {i} to branch {j}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            model.addConstr(x[i, j] <= d[j] * y[i], name=f'open_{i}_{j}')
    model.optimize()
    return model
m = solve_problem()