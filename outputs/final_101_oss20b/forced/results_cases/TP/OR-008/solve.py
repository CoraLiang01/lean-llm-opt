CSVQA_DATA = {'query': '“FreshMart,” a supermarket chain, operates several warehouses that distribute fresh produce to its various retail locations. The daily demand for each store is outlined in “customer_demand.csv,” while the available supply capacity of each warehouse is provided in “supply_capacity.csv.” The transportation cost per unit of produce from each warehouse to each store is recorded in “transportation_costs.csv.” The goal is to determine the optimal amount of fresh produce to be shipped from each warehouse to each store, ensuring that all store demands are met without exceeding the warehouse capacities, while minimizing the total transportation cost.', 'relationships': [{'column_axis': {'id_column': 'Customers', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Suppliers', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['Customers', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'Customers': 'Customer1', 'demand': '70'}}, {'data_row_number': 1, 'values': {'Customers': 'Customer2', 'demand': '80'}}, {'data_row_number': 2, 'values': {'Customers': 'Customer3', 'demand': '60'}}, {'data_row_number': 3, 'values': {'Customers': 'Customer4', 'demand': '90'}}, {'data_row_number': 4, 'values': {'Customers': 'Customer5', 'demand': '85'}}, {'data_row_number': 5, 'values': {'Customers': 'Customer6', 'demand': '95'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Suppliers', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Suppliers': 'Supplier1', 'supply_capacity': '200'}}, {'data_row_number': 1, 'values': {'Suppliers': 'Supplier2', 'supply_capacity': '250'}}, {'data_row_number': 2, 'values': {'Suppliers': 'Supplier3', 'supply_capacity': '230'}}, {'data_row_number': 3, 'values': {'Suppliers': 'Supplier4', 'supply_capacity': '220'}}, {'data_row_number': 4, 'values': {'Suppliers': 'Supplier5', 'supply_capacity': '210'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Customer1': '2', 'Customer2': '3', 'Customer3': '1', 'Customer4': '2', 'Customer5': '3', 'Customer6': '2', 'Unnamed: 0': 'Supplier1'}}, {'data_row_number': 1, 'values': {'Customer1': '1', 'Customer2': '2', 'Customer3': '3', 'Customer4': '2', 'Customer5': '3', 'Customer6': '2', 'Unnamed: 0': 'Supplier2'}}, {'data_row_number': 2, 'values': {'Customer1': '3', 'Customer2': '1', 'Customer3': '2', 'Customer4': '3', 'Customer5': '2', 'Customer6': '3', 'Unnamed: 0': 'Supplier3'}}, {'data_row_number': 3, 'values': {'Customer1': '2', 'Customer2': '3', 'Customer3': '2', 'Customer4': '1', 'Customer5': '3', 'Customer6': '4', 'Unnamed: 0': 'Supplier4'}}, {'data_row_number': 4, 'values': {'Customer1': '3', 'Customer2': '2', 'Customer3': '3', 'Customer4': '3', 'Customer5': '2', 'Customer6': '3', 'Unnamed: 0': 'Supplier5'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    supplier_table = get_table('file_1_view_0')
    customer_table = get_table('file_0_view_0')
    I = [rec['values']['Suppliers'] for rec in supplier_table['records']]
    J = [rec['values']['Customers'] for rec in customer_table['records']]
    demand = {}
    for rec in customer_table['records']:
        cust = rec['values']['Customers']
        val = rec['values']['demand']
        try:
            demand[cust] = float(val)
        except ValueError:
            raise ValueError(f'Invalid demand value for {cust}: {val}')
    supply_capacity = {}
    for rec in supplier_table['records']:
        sup = rec['values']['Suppliers']
        val = rec['values']['supply_capacity']
        try:
            supply_capacity[sup] = float(val)
        except ValueError:
            raise ValueError(f'Invalid supply capacity for {sup}: {val}')
    cost_table = get_table('file_2_view_0')
    cost = {}
    for rec in cost_table['records']:
        sup = rec['values']['Unnamed: 0']
        if sup not in cost:
            cost[sup] = {}
        for cust in J:
            if cust not in rec['values']:
                raise ValueError(f'Missing cost for supplier {sup} to customer {cust}')
            val = rec['values'][cust]
            try:
                cost[sup][cust] = float(val)
            except ValueError:
                raise ValueError(f'Invalid cost value for {sup}->{cust}: {val}')
    for sup in I:
        if sup not in supply_capacity:
            raise ValueError(f'Supply capacity missing for supplier {sup}')
        if sup not in cost:
            raise ValueError(f'Cost row missing for supplier {sup}')
        for cust in J:
            if cust not in cost[sup]:
                raise ValueError(f'Cost missing for supplier {sup} to customer {cust}')
    for cust in J:
        if cust not in demand:
            raise ValueError(f'Demand missing for customer {cust}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = model.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    model.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) <= supply_capacity[i] for i in I), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
