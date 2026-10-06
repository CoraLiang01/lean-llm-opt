import gurobipy as gp
import pandas as pd
import numpy as np
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'

def solve_problem():
    df1 = pd.read_csv(path_36_1, sep=',')
    df2 = pd.read_csv(path_36_2, sep=',')
    df3 = pd.read_csv(path_36_3, sep=',')
    product_cols = [col for col in df1.columns if col.startswith('A')]
    products = product_cols.copy()

    def get_row(df, row_label):
        row = df[df['Product'].str.strip().casefold() == row_label.strip().casefold()]
        if row.empty:
            raise KeyError(f"Row '{row_label}' not found in file.")
        return row.iloc[0]
    max_demand_row = get_row(df1, 'Maximum Demand (100 kg units)')
    selling_price_row = get_row(df1, 'Selling Price ($/100 kg)')
    prod_cost_row = get_row(df1, 'Production Cost ($/100 kg)')
    quota_row = get_row(df1, 'Production Quota (max per day)')
    activation_cost_row = get_row(df2, 'Activation Cost ($)')
    min_batch_row = get_row(df3, 'Minimum Batch Size (100 kg units)')
    max_demand = {p: float(max_demand_row[p]) for p in products}
    selling_price = {p: float(selling_price_row[p]) for p in products}
    prod_cost = {p: float(prod_cost_row[p]) for p in products}
    quota = {p: float(quota_row[p]) for p in products}
    activation_cost = {p: float(activation_cost_row[p]) for p in products}
    min_batch = {p: int(min_batch_row[p]) for p in products}
    m = gp.Model('MonthlyProductionPlan')
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='x')
    y = m.addVars(products, vtype=gp.GRB.BINARY, name='y')
    m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[p] <= max_demand[p] for p in products), name='DemandLimit')
    m.addConstrs((x[p] <= 22 * quota[p] for p in products), name='QuotaLimit')
    m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='MinBatch')
    m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='ActivationLink')
    m.addConstr(gp.quicksum((x[p] / quota[p] for p in products)) <= 22, name='TotalProductionDays')
    m.optimize()
    return m
m = solve_problem()