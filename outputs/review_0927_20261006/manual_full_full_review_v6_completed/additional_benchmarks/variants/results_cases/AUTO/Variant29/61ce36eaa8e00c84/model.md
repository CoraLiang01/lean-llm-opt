[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (total unit benefits plus bundle bonuses, minus item and category activation fees), subject to storage, staff time, and energy resource limits, category quantity bounds, item-level order bounds, incompatibility, and prerequisite/bundle logic.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: all rows in the 'item' table (item_ref).
    - Categories: all rows in the 'category' table (category).
    - Resources: all rows in the 'usage' and 'capacity_ledger' tables (resource).
    - Incompatible pairs: all rows in the 'incompatible' table (item_a, item_b).
    - Prerequisite pairs: all rows in the 'requires' table (item_ref, prerequisite_ref).
    - Bundles: all rows in the 'bundle' table (item_a, item_b).
4.  **Define Decision Variables:**
    - `q[i]` = integer quantity ordered of item i (item_ref). Type: GRB.INTEGER, domain: 0 or [minimum_lot, maximum_order] if selected.
    - `z[i]` = binary variable: 1 if item i is ordered (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `g[c]` = binary variable: 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    - `b[a,b]` = binary variable: 1 if both items a and b in bundle (a,b) are ordered, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - 'unit_benefit_cents' from 'item' table (per-unit benefit for each item).
        - 'item_fee_cents' from 'item' table (fixed fee per item if ordered).
        - 'activation_fee_cents' from 'category' table (fixed fee per category if any item in category is ordered).
        - 'bonus_cents' from 'bundle' table (bonus if both items in bundle are ordered).
    - Constraint coefficients:
        - 'amount' and 'unit' from 'usage' table (per-unit resource use for each item).
        - 'amount' and 'unit' from 'capacity_ledger' table (signed resource capacity).
    - Constraint RHS:
        - Resource limits: sum of 'amount' for each resource in 'capacity_ledger', converted to consistent units.
        - Category quantity bounds: 'minimum_quantity' and 'maximum_quantity' from 'category' table.
        - Item order bounds: 'minimum_lot' and 'maximum_order' from 'item' table.
    - Logical relationships:
        - 'authorized' from 'item' table (only items with authorized=1 are eligible).
        - Incompatibility: 'incompatible' table.
        - Prerequisites: 'requires' table.
        - Bundles: 'bundle' table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    - Sum over all authorized items: (unit_benefit_cents[i] * q[i]) 
    - Plus sum over all bundles: (bonus_cents[a,b] * b[a,b])
    - Minus sum over all items: (item_fee_cents[i] * z[i])
    - Minus sum over all categories: (activation_fee_cents[c] * g[c])
7.  **Formulate Constraints:**
    - Constraint 1 (Authorization and Order Bounds): For each item i, q[i] = 0 if authorized[i] ≠ 1; else q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (enforced via z[i]: minimum_lot[i]*z[i] ≤ q[i] ≤ maximum_order[i]*z[i], q[i]=0 iff z[i]=0).
    - Constraint 2 (Resource Limits): For each resource r, sum over items i of (usage_amount[i,r] * q[i] * unit_conversion_factor) ≤ total_capacity[r], where usage_amount and total_capacity are converted to common units (e.g., ml, minute, wh).
    - Constraint 3 (Category Quantity Bounds): For each category c, sum over items i in c of q[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    - Constraint 4 (Category Activation): For each category c, g[c] ≥ z[i] for all items i in c; g[c] = 1 iff any item in c is ordered.
    - Constraint 5 (Item Activation): For each item i, z[i] = 1 iff q[i] > 0; z[i] ∈ {0,1}.
    - Constraint 6 (Incompatibility): For each incompatible pair (a,b), z[a] + z[b] ≤ 1.
    - Constraint 7 (Prerequisites): For each (i, prereq) in 'requires', z[i] ≤ z[prereq] (if i is ordered, its prerequisite must also be ordered).
    - Constraint 8 (Bundles): For each bundle (a,b), b[a,b] ≤ z[a], b[a,b] ≤ z[b], b[a,b] ≥ z[a] + z[b] - 1 (b[a,b]=1 iff both a and b are ordered; only then is the bonus earned).
[Abstract Model Plan END]