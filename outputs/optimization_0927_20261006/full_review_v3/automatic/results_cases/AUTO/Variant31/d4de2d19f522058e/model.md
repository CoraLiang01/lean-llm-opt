[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints, using all supplied table rows directly.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and combinatorial (logic) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table.
    - Categories from the 'category' table.
    - Resources from the 'capacity_ledger' and 'usage' tables.
    - Bundles from the 'bundle' table.
    - Incompatible pairs from the 'incompatible' table.
    - Requires pairs from the 'requires' table.
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity ordered of item i (from 'item_ref'). Type: GRB.INTEGER.
    - `z[i]` = 1 if item i is selected (i.e., q[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    - `w[c]` = 1 if any item in category c is selected (i.e., sum over i in c of z[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = 1 if both items in bundle are selected (i.e., z[item_a] ≥ 1 and z[item_b] ≥ 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - 'unit_benefit_cents' (per-unit benefit, from 'item' table).
        - 'item_fee_cents' (fixed per-item fee, from 'item' table).
        - 'activation_fee_cents' (fixed per-category fee, from 'category' table).
        - 'bonus_cents' (per-bundle bonus, from 'bundle' table).
    - Constraint coefficients:
        - 'amount' (per-unit resource usage, from 'usage' table).
        - 'amount' (resource limits, from 'capacity_ledger' table: sum of 'opening' and 'reservation' for each resource).
        - 'minimum_lot', 'maximum_order' (per-item quantity bounds, from 'item' table).
        - 'authorized' (item eligibility, from 'item' table).
        - 'minimum_quantity', 'maximum_quantity' (per-category quantity bounds, from 'category' table).
        - Incompatible pairs (from 'incompatible' table).
        - Requires pairs (from 'requires' table).
    - Index mappings:
        - Item-to-category (from 'item' table).
        - Item-to-resource usage (from 'usage' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        - Sum over items: (unit_benefit_cents[i] * q[i]) 
        - Minus sum over items: (item_fee_cents[i] * z[i]) [charged once per item if any quantity is ordered]
        - Minus sum over categories: (activation_fee_cents[c] * w[c]) [charged once per category if any item in c is ordered]
        - Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [awarded once if both items in bundle are selected]
7.  **Formulate Constraints:**
    - Resource Capacity: For each resource r, sum over items i of (usage_amount[i, r] * q[i]) ≤ total_capacity[r] (from 'capacity_ledger').
    - Item Authorization: For each item i, q[i] = 0 if authorized[i] = 0; else, q[i] ∈ [minimum_lot[i], maximum_order[i]] ∪ {0}.
    - Item Selection Linking: For each item i, z[i] = 1 if q[i] ≥ 1; z[i] = 0 if q[i] = 0.
    - Category Activation Linking: For each category c, w[c] = 1 if any z[i] = 1 for i in c; w[c] = 0 otherwise.
    - Category Quantity Bounds: For each category c, sum over i in c of q[i] ∈ [minimum_quantity[c], maximum_quantity[c]] (unconditional).
    - Incompatibility: For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    - Requires Dependencies: For each requires pair (i, prereq), q[i] ≥ 1 ⇒ q[prereq] ≥ 1.
    - Bundle Bonus Linking: For each bundle (item_a, item_b), b[bundle] = 1 if z[item_a] = 1 and z[item_b] = 1; b[bundle] = 0 otherwise.
    - Integrality: All q[i] are integer, all z[i], w[c], b[bundle] are binary.
[Abstract Model Plan END]