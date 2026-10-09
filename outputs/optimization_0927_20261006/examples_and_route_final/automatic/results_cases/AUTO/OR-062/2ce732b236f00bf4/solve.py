import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['Customer'].tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['Customer']).strip()
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
suppliers = [str(s).strip() for s in fixed_cost_df['Unnamed: 0'].tolist()]
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed cost for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
transport_df = transport_df.rename(columns={'Unnamed: 0': 'Supplier'})
transport_df['Supplier'] = transport_df['Supplier'].apply(lambda x: str(x).strip())
transport_customers = [c for c in transport_df.columns if c != 'Supplier']

def normalize_name(name):
    return re.sub('\\s+', '', name).strip().casefold()
customer_map = {}
for cust in customers:
    norm_cust = normalize_name(cust)
    found = None
    for tc in transport_customers:
        if normalize_name(tc) == norm_cust:
            found = tc
            break
    if found is None:
        raise KeyError(f"Customer '{cust}' from demand.csv not found in transportation_costs.csv columns.")
    customer_map[cust] = found
transport_suppliers = [str(s).strip() for s in transport_df['Supplier'].tolist()]
supplier_map = {}
for sup in suppliers:
    norm_sup = normalize_name(sup)
    found = None
    for ts in transport_suppliers:
        if normalize_name(ts) == norm_sup:
            found = ts
            break
    if found is None:
        raise KeyError(f"Supplier '{sup}' from fixed_cost.csv not found in transportation_costs.csv rows.")
    supplier_map[sup] = found
transport_cost_dict = {}
for sup in suppliers:
    tsup = supplier_map[sup]
    row = transport_df[transport_df['Supplier'] == tsup]
    if row.empty:
        raise KeyError(f"Supplier '{tsup}' not found in transportation_costs.csv rows.")
    row = row.iloc[0]
    for cust in customers:
        tcust = customer_map[cust]
        try:
            cost = float(row[tcust])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{sup}', customer '{cust}': {row[tcust]}")
        transport_cost_dict[sup, cust] = cost
supplier_set = suppliers
customer_set = customers
bigM = sum((demand_dict[cust] for cust in customer_set))
m = gp.Model('Iowa_Liquor_UFLP')
y_vars = m.addVars(supplier_set, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(supplier_set, customer_set, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[sup] * y_vars[sup] for sup in supplier_set))
transport_cost_expr = gp.quicksum((transport_cost_dict[sup, cust] * x_vars[sup, cust] for sup in supplier_set for cust in customer_set))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for cust in customer_set:
    m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in supplier_set)) == demand_dict[cust], name=f'demand_{cust}')
for sup in supplier_set:
    for cust in customer_set:
        m.addConstr(x_vars[sup, cust] <= bigM * y_vars[sup], name=f'link_{sup}_{cust}')
m.optimize()