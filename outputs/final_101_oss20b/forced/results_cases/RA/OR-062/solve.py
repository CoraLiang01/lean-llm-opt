CSVQA_DATA = {'query': 'The Iowa Department of Commerce requires that any store selling alcohol in bottled form for off-premises consumption must hold a Class “E” liquor license, a typical arrangement for most state liquor regulatory authorities. All alcohol sales from stores registered with the Iowa Department of Commerce are recorded in the department’s system, which is publicly released as open data by the State of Iowa. Several suppliers located in different cities can provide the necessary liquor products to these licensed stores. Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in “fixed_cost.csv.” The Department needs to source a unit of each liquor product for the stores from these suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for all liquor products across all licensed stores is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.', 'relationships': [], 'tables': [{'columns': ['Customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'Customer': 'Customer_1', 'demand': '2397'}}, {'data_row_number': 1, 'values': {'Customer': 'Customer_2', 'demand': '1889'}}, {'data_row_number': 2, 'values': {'Customer': 'Customer_3', 'demand': '2518'}}, {'data_row_number': 3, 'values': {'Customer': 'Customer_4', 'demand': '3218'}}, {'data_row_number': 4, 'values': {'Customer': 'Customer_5', 'demand': '1813'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'MOUNT AYR', 'fixed_costs': '96.58'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'WAUKEE', 'fixed_costs': '94.06'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'WAVERLY', 'fixed_costs': '94.37'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'PELLA', 'fixed_costs': '82.88'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'BANCROFT': '1685.53', 'CLARINDA': '694.6799999999999', 'FORT MADISON': '17.48', 'SIOUX CITY': '20.07', 'TOLEDO': '199.02', 'Unnamed: 0': 'MOUNT AYR'}}, {'data_row_number': 1, 'values': {'BANCROFT': '90.69', 'CLARINDA': '15.13', 'FORT MADISON': '1.5', 'SIOUX CITY': '1.43', 'TOLEDO': '27.88', 'Unnamed: 0': 'WAUKEE'}}, {'data_row_number': 2, 'values': {'BANCROFT': '78.73', 'CLARINDA': '2.34', 'FORT MADISON': '349.34', 'SIOUX CITY': '246.6', 'TOLEDO': '41.3', 'Unnamed: 0': 'WAVERLY'}}, {'data_row_number': 3, 'values': {'BANCROFT': '38.93', 'CLARINDA': '1181.6', 'FORT MADISON': '1458.53', 'SIOUX CITY': '1646.36', 'TOLEDO': '1924.55', 'Unnamed: 0': 'PELLA'}}, {'data_row_number': 4, 'values': {'BANCROFT': '103.84', 'CLARINDA': '1030.8', 'FORT MADISON': '43.48', 'SIOUX CITY': '932.4299999999999', 'TOLEDO': '55.39', 'Unnamed: 0': 'DES MOINES'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    demand_table = tables['file_0_view_0']
    J = []
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['Customer']
        J.append(cust)
        try:
            demand[cust] = int(rec['values']['demand'])
        except ValueError:
            raise ValueError(f'Invalid demand value for customer {cust}')
    fixed_table = tables['file_1_view_0']
    I = []
    fixed_cost = {}
    for rec in fixed_table['records']:
        sup = rec['values']['Unnamed: 0']
        I.append(sup)
        try:
            fixed_cost[sup] = float(rec['values']['fixed_costs'])
        except ValueError:
            raise ValueError(f'Invalid fixed cost for supplier {sup}')
    trans_table = tables['file_2_view_0']
    trans_cost = {}
    for rec in trans_table['records']:
        sup = rec['values']['Unnamed: 0']
        for cust in J:
            if cust not in rec['values']:
                raise ValueError(f'Missing transportation cost for supplier {sup} to customer {cust}')
            try:
                trans_cost[sup, cust] = float(rec['values'][cust])
            except ValueError:
                raise ValueError(f'Invalid transportation cost for supplier {sup} to customer {cust}')
    for sup in I:
        if sup not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
    for cust in J:
        if cust not in demand:
            raise ValueError(f'Missing demand for customer {cust}')
    for sup in I:
        for cust in J:
            if (sup, cust) not in trans_cost:
                raise ValueError(f'Missing transportation cost for supplier {sup} to customer {cust}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in I)) + gurobipy.quicksum((trans_cost[i, j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= demand[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= demand[j] * y[i], name=f'activate_{i}_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
