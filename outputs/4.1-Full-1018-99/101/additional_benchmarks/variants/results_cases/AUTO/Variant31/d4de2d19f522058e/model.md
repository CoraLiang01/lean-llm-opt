[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for the next delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource capacity, category quantity limits, item authorization and lot/order bounds, incompatibility and dependency (requires) constraints, and bundle bonus logic. All constraints and fees/bonuses must be respected exactly as described.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility and dependency) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table (item_ref).
    - Categories from the 'category' table (category).
    - Resources from the 'resource' field in 'usage' and 'capacity_ledger' tables.
    - Bundles (pairs of items) from the 'bundle' table.
    - Incompatible pairs from the 'incompatible' table.
    - Requires pairs from the 'requires' table.
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity of item i selected for delivery (i in Items). Type: GRB.INTEGER.
    - `z[i]` = Binary variable: 1 if item i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = Binary variable: 1 if any item in category c is selected, 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - 'unit_benefit_cents' (per-unit benefit for each item, from 'item' table).
        - 'item_fee_cents' (fixed fee per item if selected, from 'item' table).
        - 'activation_fee_cents' (fixed fee per category if any item in the category is selected, from 'category' table).
        - 'bonus_cents' (bonus per bundle, from 'bundle' table).
    - Constraint coefficients:
        - 'amount' (resource usage per unit per item, from 'usage' table).
        - 'amount' (resource available, sum of 'opening' and 'reservation' in 'capacity_ledger' table).
        - 'minimum_lot', 'maximum_order' (per-item quantity bounds, from 'item' table).
        - 'authorized' (item eligibility, from 'item' table).
        - 'minimum_quantity', 'maximum_quantity' (per-category aggregate quantity bounds, from 'category' table).
        - Incompatible pairs (from 'incompatible' table).
        - Requires pairs (from 'requires' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * q[i]) 
    - Minus sum over all items: (item_fee_cents[i] * z[i]) [charged once per item if any quantity is selected]
    - Minus sum over all categories: (activation_fee_cents[c] * y[c]) [charged once per category if any item in the category is selected]
    - Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items are selected]
7.  **Formulate Constraints:**
    - **Item Authorization and Quantity Bounds:** For each item i:
        - If authorized[i] == 1: minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; q[i] ≥ 0 and integer; z[i] ∈ {0,1}.
        - If authorized[i] == 0: q[i] = 0; z[i] = 0.
    - **Category Quantity Limits:** For each category c:
        - Sum over items in category c: minimum_quantity[c] ≤ Σ q[i in c] ≤ maximum_quantity[c].
        - y[c] = 1 if any z[i in c] = 1; y[c] = 0 otherwise (enforced by: z[i] ≤ y[c] for all i in c; y[c] ≤ Σ z[i in c]).
    - **Resource Capacity Constraints:** For each resource r:
        - Σ (usage_amount[i,r] * q[i]) ≤ total_capacity[r], where total_capacity[r] = sum of 'amount' for 'opening' and 'reservation' entries in 'capacity_ledger' for resource r.
    - **Incompatibility Constraints:** For each incompatible pair (i, j):
        - z[i] + z[j] ≤ 1 (cannot select both items in an incompatible pair).
    - **Requires Constraints:** For each requires pair (i, prereq):
        - z[i] ≤ z[prereq] (can only select i if prereq is also selected).
        - q[i] > 0 ⇒ q[prereq] > 0 (enforced via z variables).
    - **Bundle Bonus Logic:** For each bundle (i_a, i_b):
        - b[bundle] ≤ z[i_a]; b[bundle] ≤ z[i_b]; b[bundle] ≥ z[i_a] + z[i_b] - 1 (b[bundle] = 1 iff both items are selected).
    - **Variable Domains:** All q[i] are integer ≥ 0; all z[i], y[c], b[bundle] are binary.
[Abstract Model Plan END]