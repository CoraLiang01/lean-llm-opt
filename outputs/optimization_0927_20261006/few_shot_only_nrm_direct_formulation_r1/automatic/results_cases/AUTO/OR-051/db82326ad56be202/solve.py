import gurobipy as gp
from gurobipy import GRB
cabinets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cabinet_capacity = {1: 400, 2: 600, 3: 500, 4: 700, 5: 450, 6: 650, 7: 550, 8: 750, 9: 480, 10: 520}
products = ['Espresso Beans', 'Colombian Roast', 'Arabica Blend', 'French Roast', 'Italian Roast', 'House Blend', 'Sumatra Coffee', 'Mocha Java', 'Hazelnut Flavor', 'Caramel Blend', 'Vanilla Flavor', 'Cappuccino Mix', 'Pumpkin Spice', 'Decaf Roast', 'Organic Roast', 'Cold Brew', 'Peruvian Blend', 'Kenyan AA']
product_value = {'Espresso Beans': 100, 'Colombian Roast': 150, 'Arabica Blend': 80, 'French Roast': 120, 'Italian Roast': 130, 'House Blend': 110, 'Sumatra Coffee': 160, 'Mocha Java': 90, 'Hazelnut Flavor': 95, 'Caramel Blend': 105, 'Vanilla Flavor': 85, 'Cappuccino Mix': 140, 'Pumpkin Spice': 75, 'Decaf Roast': 60, 'Organic Roast': 170, 'Cold Brew': 115, 'Peruvian Blend': 155, 'Kenyan AA': 125}
product_weight = {'Espresso Beans': 1.0, 'Colombian Roast': 1.5, 'Arabica Blend': 1.2, 'French Roast': 1.3, 'Italian Roast': 1.4, 'House Blend': 1.1, 'Sumatra Coffee': 1.8, 'Mocha Java': 1.2, 'Hazelnut Flavor': 1.0, 'Caramel Blend': 1.3, 'Vanilla Flavor': 1.2, 'Cappuccino Mix': 1.5, 'Pumpkin Spice': 1.1, 'Decaf Roast': 1.0, 'Organic Roast': 1.6, 'Cold Brew': 1.4, 'Peruvian Blend': 1.7, 'Kenyan AA': 1.3}
if set(cabinet_capacity.keys()) != set(cabinets):
    raise ValueError('Cabinet capacity data missing for some cabinets.')
if set(product_value.keys()) != set(products):
    raise ValueError('Product value data missing for some products.')
if set(product_weight.keys()) != set(products):
    raise ValueError('Product weight data missing for some products.')
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinets, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_value[j] * x_vars[i, j] for i in cabinets for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weight[j] * x_vars[i, j] for j in products)) <= cabinet_capacity[i] for i in cabinets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')