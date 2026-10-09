[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints, using all supplied table rows directly.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and combinatorial (logic) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the item table.
    - Categories from the category table.
    - Resources from the capacity_ledger and usage tables.
    - Bundles (item pairs) from the bundle table.
    - Incompatible pairs from the incompatible table.
    - Requires pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected for delivery (must be zero if unauthorized; otherwise between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `y[i]` = Binary variable: 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both y[i_a] and y[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (per unit, from item table).
        -   `item_fee_cents` (fixed per item if any units selected, from item table).
        -   `activation_fee_cents` (fixed per category if any item in category selected, from category table).
        -   `bonus_cents` (per bundle, from bundle table).
    -   Constraint coefficients:
        -   `usage` (resource usage per unit per item, from usage table).
        -   `capacity_ledger` (resource available, sum of opening and reservation, from capacity_ledger table).
        -   `minimum_lot`, `maximum_order` (per item, from item table).
        -   `minimum_quantity`, `maximum_quantity` (per category, from category table).
        -   `authorized` (per item, from item table).
        -   Incompatible pairs (from incompatible table).
        -   Requires pairs (from requires table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over items: (item_fee_cents[i] * y[i]) [charged once per item if any units selected]
    -   Minus sum over categories: (activation_fee_cents[c] * z[c]) [charged once per category if any item in category selected]
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items selected]
7.  **Formulate Constraints:**
    -   Resource Capacity: For each resource, total usage across all items does not exceed the signed total in capacity_ledger (sum of opening and reservation for each resource).
        -   sum over items: (usage[i, r] * x[i]) ≤ capacity_ledger[r]
    -   Item Authorization and Lot/Order Bounds: For each item,
        -   If authorized[i] = 1: minimum_lot[i] ≤ x[i] ≤ maximum_order[i]
        -   If authorized[i] = 0: x[i] = 0
    -   Item Selection Linking: For each item,
        -   y[i] = 1 if x[i] ≥ 1, y[i] = 0 if x[i] = 0 (enforced via standard linking constraints)
    -   Category Quantity Limits: For each category,
        -   minimum_quantity[c] ≤ sum over items in c: x[i] ≤ maximum_quantity[c]
    -   Category Activation Linking: For each category,
        -   z[c] = 1 if any x[i] > 0 for items in c, z[c] = 0 otherwise (enforced via linking constraints)
    -   Incompatibility: For each incompatible pair (i, j),
        -   y[i] + y[j] ≤ 1 (cannot select both items)
    -   Requires Dependencies: For each requires pair (i, prereq),
        -   y[i] ≤ y[prereq] (can only select i if prerequisite is also selected)
    -   Bundle Bonuses: For each bundle (i_a, i_b),
        -   b[bundle] ≤ y[i_a], b[bundle] ≤ y[i_b], b[bundle] ≥ y[i_a] + y[i_b] - 1 (b[bundle] = 1 iff both items selected)
[Abstract Model Plan END]