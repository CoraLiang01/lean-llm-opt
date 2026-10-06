import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def filter_valid(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        return df

    def filter_valid_bundle(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['item_a'].notnull() & df['item_b'].notnull()]
        return df

    def filter_valid_usage(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['item_ref'].notnull() & df['resource'].notnull()]
        return df

    def filter_valid_capacity_ledger(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['resource'].notnull()]
        return df

    def filter_valid_fx(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['currency', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['currency'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['currency'].notnull()]
        return df

    def filter_valid_item_fee(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['item_ref'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['item_ref'].notnull()]
        return df

    def filter_valid_category(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['category'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['category'].notnull()]
        return df

    def filter_valid_item(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['item_ref'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['item_ref'].notnull()]
        return df

    def filter_valid_incompatible(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['item_a'].notnull() & df['item_b'].notnull()]
        return df

    def filter_valid_requires(df, dealership_id, table, as_of_date):
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['table'].str.casefold() == table.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        df = df[df['action'].str.casefold() != 'delete']
        df = df[df['item_ref'].notnull() & df['prerequisite_ref'].notnull()]
        return df
    table_map = {}
    for df in dfs:
        if 'table' in df.columns:
            t = df['table'].iloc[0].strip().casefold()
            if t not in table_map:
                table_map[t] = []
            table_map[t].append(df)
    as_of_date = '2026-05-07'
    dealership_id = 'OSLO_NEW_CARS'
    benefit_dfs = []
    for t in ['benefit']:
        for df in table_map.get(t, []):
            benefit_dfs.append(df)
    benefit_df = pd.concat(benefit_dfs, ignore_index=True)
    benefit_df = filter_valid(benefit_df, dealership_id, 'benefit', as_of_date)
    bundle_dfs = []
    for t in ['bundle']:
        for df in table_map.get(t, []):
            bundle_dfs.append(df)
    bundle_df = pd.concat(bundle_dfs, ignore_index=True)
    bundle_df = filter_valid_bundle(bundle_df, dealership_id, 'bundle', as_of_date)
    cap_dfs = []
    for t in ['capacity_ledger']:
        for df in table_map.get(t, []):
            cap_dfs.append(df)
    cap_df = pd.concat(cap_dfs, ignore_index=True)
    cap_df = filter_valid_capacity_ledger(cap_df, dealership_id, 'capacity_ledger', as_of_date)
    cat_dfs = []
    for t in ['category']:
        for df in table_map.get(t, []):
            cat_dfs.append(df)
    cat_df = pd.concat(cat_dfs, ignore_index=True)
    cat_df = filter_valid_category(cat_df, dealership_id, 'category', as_of_date)
    fx_dfs = []
    for t in ['fx']:
        for df in table_map.get(t, []):
            fx_dfs.append(df)
    fx_df = pd.concat(fx_dfs, ignore_index=True)
    fx_df = filter_valid_fx(fx_df, dealership_id, 'fx', as_of_date)
    item_fee_dfs = []
    for t in ['item_fee']:
        for df in table_map.get(t, []):
            item_fee_dfs.append(df)
    item_fee_df = pd.concat(item_fee_dfs, ignore_index=True)
    item_fee_df = filter_valid_item_fee(item_fee_df, dealership_id, 'item_fee', as_of_date)
    item_dfs = []
    for t in ['item']:
        for df in table_map.get(t, []):
            item_dfs.append(df)
    item_df = pd.concat(item_dfs, ignore_index=True)
    item_df = filter_valid_item(item_df, dealership_id, 'item', as_of_date)
    incompatible_dfs = []
    for t in ['incompatible']:
        for df in table_map.get(t, []):
            incompatible_dfs.append(df)
    incompatible_df = pd.concat(incompatible_dfs, ignore_index=True)
    incompatible_df = filter_valid_incompatible(incompatible_df, dealership_id, 'incompatible', as_of_date)
    requires_dfs = []
    for t in ['requires']:
        for df in table_map.get(t, []):
            requires_dfs.append(df)
    requires_df = pd.concat(requires_dfs, ignore_index=True)
    requires_df = filter_valid_requires(requires_df, dealership_id, 'requires', as_of_date)
    usage_dfs = []
    for t in ['usage']:
        for df in table_map.get(t, []):
            usage_dfs.append(df)
    usage_df = pd.concat(usage_dfs, ignore_index=True)
    usage_df = filter_valid_usage(usage_df, dealership_id, 'usage', as_of_date)
    items = sorted(item_df['item_ref'].unique())
    categories = sorted(cat_df['category'].unique())
    resources = sorted(cap_df['resource'].unique())
    bundles = []
    for (_, row) in bundle_df.iterrows():
        bundles.append((row['item_a'], row['item_b']))
    incompat_pairs = []
    for (_, row) in incompatible_df.iterrows():
        incompat_pairs.append((row['item_a'], row['item_b']))
    requires_pairs = []
    for (_, row) in requires_df.iterrows():
        requires_pairs.append((row['item_ref'], row['prerequisite_ref']))
    fx_dict = {}
    for (_, row) in fx_df.iterrows():
        c = str(row['currency'])
        fx_dict[c] = (float(row['usd_cents_numerator']), float(row['denominator']))
    benefit_per_item = {}
    for i in items:
        df_i = benefit_df[benefit_df['item_ref'] == i]
        total = 0.0
        for (_, row) in df_i.iterrows():
            amt = float(row['amount'])
            curr = str(row['currency'])
            if curr not in fx_dict:
                raise ValueError(f'Missing FX rate for currency {curr} as of {as_of_date}')
            (num, denom) = fx_dict[curr]
            total += amt * num / denom
        benefit_per_item[i] = total
    item_fee_dict = {}
    for i in items:
        df_i = item_fee_df[item_fee_df['item_ref'] == i]
        if not df_i.empty:
            item_fee_dict[i] = float(df_i.iloc[0]['activation_fee_cents'])
        else:
            item_fee_dict[i] = 0.0
    min_g = {}
    max_g = {}
    cat_fee = {}
    for g in categories:
        df_g = cat_df[cat_df['category'] == g]
        if not df_g.empty:
            min_g[g] = float(df_g.iloc[0]['minimum_quantity'])
            max_g[g] = float(df_g.iloc[0]['maximum_quantity'])
            cat_fee[g] = float(df_g.iloc[0]['activation_fee_cents'])
        else:
            raise ValueError(f'Missing category info for {g}')
    auth = {}
    min_lot = {}
    max_order = {}
    cat_of = {}
    for i in items:
        df_i = item_df[item_df['item_ref'] == i]
        if not df_i.empty:
            row = df_i.iloc[0]
            auth[i] = int(row['authorized'])
            min_lot[i] = int(row['minimum_lot'])
            max_order[i] = int(row['maximum_order'])
            cat_of[i] = row['category']
        else:
            raise ValueError(f'Missing item info for {i}')
    usage_dict = {}
    for (_, row) in usage_df.iterrows():
        i = row['item_ref']
        r = row['resource']
        amt = float(row['amount'])
        unit = str(row['unit']).casefold()
        if r == 'space':
            if unit == 'liter':
                amt = amt * 1000.0
            elif unit == 'ml':
                pass
            else:
                raise ValueError(f'Unknown unit for space: {unit}')
        elif r == 'power':
            if unit == 'kwh':
                amt = amt * 1000.0
            elif unit == 'wh':
                pass
            else:
                raise ValueError(f'Unknown unit for power: {unit}')
        elif r == 'labor':
            if unit == 'hour':
                amt = amt * 60.0
            elif unit == 'minute':
                pass
            else:
                raise ValueError(f'Unknown unit for labor: {unit}')
        else:
            raise ValueError(f'Unknown resource: {r}')
        usage_dict[i, r] = amt
    cap = {}
    for r in resources:
        df_r = cap_df[cap_df['resource'] == r]
        total = 0.0
        for (_, row) in df_r.iterrows():
            amt = float(row['amount'])
            unit = str(row['unit']).casefold()
            if r == 'space':
                if unit == 'liter':
                    amt = amt * 1000.0
                elif unit == 'ml':
                    pass
                else:
                    raise ValueError(f'Unknown unit for space: {unit}')
            elif r == 'power':
                if unit == 'kwh':
                    amt = amt * 1000.0
                elif unit == 'wh':
                    pass
                else:
                    raise ValueError(f'Unknown unit for power: {unit}')
            elif r == 'labor':
                if unit == 'hour':
                    amt = amt * 60.0
                elif unit == 'minute':
                    pass
                else:
                    raise ValueError(f'Unknown unit for labor: {unit}')
            else:
                raise ValueError(f'Unknown resource: {r}')
            total += amt
        cap[r] = total
    bonus = {}
    for (_, row) in bundle_df.iterrows():
        a = row['item_a']
        b = row['item_b']
        bonus[a, b] = float(row['bonus_cents'])
    for i in items:
        if i not in benefit_per_item:
            raise ValueError(f'Missing benefit for item {i}')
        if i not in item_fee_dict:
            raise ValueError(f'Missing item_fee for item {i}')
        if i not in auth:
            raise ValueError(f'Missing auth for item {i}')
        if i not in min_lot:
            raise ValueError(f'Missing min_lot for item {i}')
        if i not in max_order:
            raise ValueError(f'Missing max_order for item {i}')
        if i not in cat_of:
            raise ValueError(f'Missing category for item {i}')
    m = gp.Model('oslo_new_cars_max_net_benefit')
    m.setParam('MIPGap', 0.0001)
    q = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if auth[i] == 0:
            m.addConstr(q[i] == 0, name=f'unauth_{i}')
            m.addConstr(z[i] == 0, name=f'unauth_z_{i}')
        else:
            m.addConstr(q[i] >= min_lot[i] * z[i], name=f'minlot_{i}')
            m.addConstr(q[i] <= max_order[i] * z[i], name=f'maxorder_{i}')
            m.addConstr(q[i] >= 0, name=f'qnonneg_{i}')
    for g in categories:
        items_in_g = [i for i in items if cat_of[i] == g]
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) >= min_g[g], name=f'catmin_{g}')
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) <= max_g[g], name=f'catmax_{g}')
    for g in categories:
        items_in_g = [i for i in items if cat_of[i] == g]
        for i in items_in_g:
            m.addConstr(y[g] >= z[i], name=f'catact_{g}_{i}')
        m.addConstr(y[g] <= gp.quicksum((z[i] for i in items_in_g)), name=f'catactsum_{g}')
    for r in resources:
        expr = gp.LinExpr()
        for i in items:
            if (i, r) in usage_dict:
                expr += usage_dict[i, r] * q[i]
        m.addConstr(expr <= cap[r], name=f'cap_{r}')
    for (i, j) in incompat_pairs:
        if i in items and j in items:
            m.addConstr(z[i] + z[j] <= 1, name=f'incompat_{i}_{j}')
    for (i, k) in requires_pairs:
        if i in items and k in items:
            m.addConstr(z[i] <= z[k], name=f'requires_{i}_{k}')
    for (a, b) in bundles:
        if a in items and b in items and (auth[a] > 0) and (auth[b] > 0):
            m.addConstr(w[a, b] <= z[a], name=f'bundle1_{a}_{b}')
            m.addConstr(w[a, b] <= z[b], name=f'bundle2_{a}_{b}')
            m.addConstr(w[a, b] >= z[a] + z[b] - 1, name=f'bundle3_{a}_{b}')
        else:
            m.addConstr(w[a, b] == 0, name=f'bundle0_{a}_{b}')
    obj = gp.LinExpr()
    obj += gp.quicksum((benefit_per_item[i] * q[i] for i in items))
    obj -= gp.quicksum((item_fee_dict[i] * z[i] for i in items))
    obj -= gp.quicksum((cat_fee[g] * y[g] for g in categories))
    obj += gp.quicksum((bonus[a, b] * w[a, b] for (a, b) in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')