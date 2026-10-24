CSVQA_DATA = {'query': "A manufacturing company plans production of a seasonal appliance over a 24-month horizon. The planning data are provided in monthly_lot_sizing.csv, including each month's demand, unit production cost, fixed setup cost, unit inventory holding cost, and ProductionCapacity. The ProductionCapacity column represents the maximum number of units that can be produced in that month; it is not an inventory storage capacity. Initial inventory is zero, unmet demand is not allowed, and the plan must end with zero inventory after month 24.\n\nFormulate a mixed-integer lot-sizing model. For each month t, define x_t as the production quantity, I_t as the ending inventory, and y_t as a binary variable equal to 1 if production is set up in month t and 0 otherwise. The objective is to minimize the total cost, including production costs, fixed setup costs, and inventory holding costs. The model should include monthly inventory-balance constraints, production-capacity/setup-linking constraints, the final zero-inventory requirement, nonnegativity constraints for production and inventory, and binary restrictions for the setup variables.", 'relationships': [], 'tables': [{'columns': ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity'], 'file_name': 'monthly_lot_sizing.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '80', 'HoldingCost': '1.1', 'Month': 'M01', 'ProductionCapacity': '420', 'ProductionCost': '18', 'SetupCost': '900'}}, {'data_row_number': 1, 'values': {'Demand': '120', 'HoldingCost': '1.2', 'Month': 'M02', 'ProductionCapacity': '360', 'ProductionCost': '20', 'SetupCost': '1100'}}, {'data_row_number': 2, 'values': {'Demand': '60', 'HoldingCost': '1.0', 'Month': 'M03', 'ProductionCapacity': '380', 'ProductionCost': '19', 'SetupCost': '950'}}, {'data_row_number': 3, 'values': {'Demand': '150', 'HoldingCost': '1.3', 'Month': 'M04', 'ProductionCapacity': '420', 'ProductionCost': '21', 'SetupCost': '1200'}}, {'data_row_number': 4, 'values': {'Demand': '90', 'HoldingCost': '1.1', 'Month': 'M05', 'ProductionCapacity': '350', 'ProductionCost': '18', 'SetupCost': '850'}}, {'data_row_number': 5, 'values': {'Demand': '110', 'HoldingCost': '1.4', 'Month': 'M06', 'ProductionCapacity': '430', 'ProductionCost': '22', 'SetupCost': '1300'}}, {'data_row_number': 6, 'values': {'Demand': '140', 'HoldingCost': '1.2', 'Month': 'M07', 'ProductionCapacity': '390', 'ProductionCost': '20', 'SetupCost': '1000'}}, {'data_row_number': 7, 'values': {'Demand': '70', 'HoldingCost': '1.0', 'Month': 'M08', 'ProductionCapacity': '360', 'ProductionCost': '19', 'SetupCost': '900'}}, {'data_row_number': 8, 'values': {'Demand': '160', 'HoldingCost': '1.5', 'Month': 'M09', 'ProductionCapacity': '440', 'ProductionCost': '23', 'SetupCost': '1400'}}, {'data_row_number': 9, 'values': {'Demand': '100', 'HoldingCost': '1.2', 'Month': 'M10', 'ProductionCapacity': '400', 'ProductionCost': '21', 'SetupCost': '1150'}}, {'data_row_number': 10, 'values': {'Demand': '130', 'HoldingCost': '1.1', 'Month': 'M11', 'ProductionCapacity': '380', 'ProductionCost': '20', 'SetupCost': '1000'}}, {'data_row_number': 11, 'values': {'Demand': '85', 'HoldingCost': '1.0', 'Month': 'M12', 'ProductionCapacity': '350', 'ProductionCost': '18', 'SetupCost': '850'}}, {'data_row_number': 12, 'values': {'Demand': '115', 'HoldingCost': '1.1', 'Month': 'M13', 'ProductionCapacity': '390', 'ProductionCost': '19', 'SetupCost': '950'}}, {'data_row_number': 13, 'values': {'Demand': '95', 'HoldingCost': '1.4', 'Month': 'M14', 'ProductionCapacity': '430', 'ProductionCost': '22', 'SetupCost': '1250'}}, {'data_row_number': 14, 'values': {'Demand': '145', 'HoldingCost': '1.2', 'Month': 'M15', 'ProductionCapacity': '410', 'ProductionCost': '21', 'SetupCost': '1150'}}, {'data_row_number': 15, 'values': {'Demand': '105', 'HoldingCost': '1.1', 'Month': 'M16', 'ProductionCapacity': '370', 'ProductionCost': '20', 'SetupCost': '1050'}}, {'data_row_number': 16, 'values': {'Demand': '75', 'HoldingCost': '1.0', 'Month': 'M17', 'ProductionCapacity': '360', 'ProductionCost': '18', 'SetupCost': '800'}}, {'data_row_number': 17, 'values': {'Demand': '155', 'HoldingCost': '1.5', 'Month': 'M18', 'ProductionCapacity': '440', 'ProductionCost': '23', 'SetupCost': '1350'}}, {'data_row_number': 18, 'values': {'Demand': '125', 'HoldingCost': '1.2', 'Month': 'M19', 'ProductionCapacity': '400', 'ProductionCost': '21', 'SetupCost': '1100'}}, {'data_row_number': 19, 'values': {'Demand': '90', 'HoldingCost': '1.1', 'Month': 'M20', 'ProductionCapacity': '360', 'ProductionCost': '19', 'SetupCost': '950'}}, {'data_row_number': 20, 'values': {'Demand': '135', 'HoldingCost': '1.3', 'Month': 'M21', 'ProductionCapacity': '420', 'ProductionCost': '22', 'SetupCost': '1200'}}, {'data_row_number': 21, 'values': {'Demand': '65', 'HoldingCost': '1.0', 'Month': 'M22', 'ProductionCapacity': '350', 'ProductionCost': '18', 'SetupCost': '850'}}, {'data_row_number': 22, 'values': {'Demand': '150', 'HoldingCost': '1.2', 'Month': 'M23', 'ProductionCapacity': '430', 'ProductionCost': '20', 'SetupCost': '1000'}}, {'data_row_number': 23, 'values': {'Demand': '110', 'HoldingCost': '0.0', 'Month': 'M24', 'ProductionCapacity': '390', 'ProductionCost': '21', 'SetupCost': '1100'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    d = {}
    c = {}
    s = {}
    h = {}
    p = {}
    for rec in records:
        vals = rec['values']
        month = vals['Month']
        d[month] = float(vals['Demand'])
        c[month] = float(vals['ProductionCost'])
        s[month] = float(vals['SetupCost'])
        h[month] = float(vals['HoldingCost'])
        p[month] = float(vals['ProductionCapacity'])
    months = sorted(d.keys(), key=lambda m: int(m[1:]))
    m = gurobipy.Model('LotSizing')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(months, lb=0, vtype=GRB.CONTINUOUS, name='')
    I = m.addVars(months, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(months, vtype=GRB.BINARY, name='')
    for idx, t in enumerate(months):
        if idx == 0:
            m.addConstr(x[t] - I[t] == d[t], name=f'inv_bal_{t}')
        else:
            prev = months[idx - 1]
            m.addConstr(I[prev] + x[t] == d[t] + I[t], name=f'inv_bal_{t}')
    for t in months:
        m.addConstr(x[t] <= p[t] * y[t], name=f'cap_link_{t}')
    m.addConstr(I[months[-1]] == 0, name='final_zero_inventory')
    obj = gurobipy.quicksum((c[t] * x[t] + s[t] * y[t] + h[t] * I[t] for t in months))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
