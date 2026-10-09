[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints. Only authorized options may be selected, and each has minimum/maximum lot sizes. Resource and category limits, incompatibilities, requires dependencies, and bundle bonuses must all be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (compatibility/dependency) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the item table (item_ref).
    - Categories from the category table (category).
    - Resources from the resource/capacity_ledger/usage tables (resource).
    - Bundles (pairs of items eligible for a bonus) from the bundle table.
    - Incompatible pairs from the incompatible table.
    - Requires dependencies from the requires table.
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity of item `i` selected for delivery (0 if not selected). Type: GRB.INTEGER.
    - `z[i]` = 1 if item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = 1 if both items in bundle `bundle` are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - `unit_benefit_cents[i]` from item table: per-unit benefit for item `i`.
        - `item_fee_cents[i]` from item table: fixed fee if any of item `i` is selected.
        - `activation_fee_cents[c]` from category table: fixed fee if any item in category `c` is selected.
        - `bonus_cents[bundle]` from bundle table: bonus if both items in bundle are selected.
    - Constraint coefficients:
        - `usage[i, r]` from usage table: amount of resource `r` used per unit of item `i`.
        - `capacity_ledger[r]` from capacity_ledger table: total available amount of resource `r` (sum of opening and reservation for each resource).
        - `minimum_lot[i]`, `maximum_order[i]` from item table: min/max allowed quantity for item `i` (if authorized).
        - `authorized[i]` from item table: 1 if item `i` is eligible, 0 otherwise.
        - `category[i]` from item table: category of item `i`.
        - `minimum_quantity[c]`, `maximum_quantity[c]` from category table: unconditional min/max total quantity for category `c`.
        - Incompatible pairs: from incompatible table (item_a, item_b).
        - Requires dependencies: from requires table (item_ref, prerequisite_ref).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (unit_benefit_cents[i] * q[i]) 
    - Minus sum over items: (item_fee_cents[i] * z[i]) [charged once per item if selected]
    - Minus sum over categories: (activation_fee_cents[c] * y[c]) [charged once per category if any item in c is selected]
    - Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items are selected]
7.  **Formulate Constraints:**
    - **Authorization and Lot Size:** For each item `i`, q[i] = 0 if authorized[i] = 0; if authorized[i] = 1, then minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i], and z[i] ∈ {0,1}.
    - **Resource Limits:** For each resource `r`, sum over items of (usage[i, r] * q[i]) ≤ total available capacity_ledger[r] (sum opening + reservation for r).
    - **Category Quantity Limits:** For each category `c`, sum over items in c of q[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c] (unconditional, regardless of selection).
    - **Category Activation:** For each category `c`, y[c] ≥ z[i] for all items i in c; y[c] = 1 iff any item in c is selected.
    - **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1 (cannot select both).
    - **Requires Dependencies:** For each (i, prereq), z[i] ≤ z[prereq] (cannot select i unless prereq is also selected with positive quantity).
    - **Bundle Bonuses:** For each bundle (i_a, i_b), b[bundle] ≤ z[i_a], b[bundle] ≤ z[i_b], b[bundle] ≥ z[i_a] + z[i_b] - 1 (b[bundle] = 1 iff both items are selected).
    - **Variable Domains:** q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; z[i], y[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]