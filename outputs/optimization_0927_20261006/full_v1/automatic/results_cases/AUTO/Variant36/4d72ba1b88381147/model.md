[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer quantities of authorized air-conditioner models to assign to storage areas at FC_EAST_HVAC, maximizing net value (benefit minus all fixed and variable fees, plus any applicable bundle bonuses), subject to area volume limits, per-category quantity bounds, incompatibility and prerequisite rules, and bundle bonuses, using only the supplied item rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, assignment, and logical (incompatibility/prerequisite) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All item_ref values from the supplied item tables (only those rows).
    - Categories (`C`): All category values from the category table.
    - Resources/Areas (`R`): All resource/location_id values from the capacity_ledger and usage tables.
    - Bundles (`B`): All bundle rows (pairs of item_refs) from the bundle table.
    - Incompatibility pairs and prerequisite pairs from their respective tables.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (must be 0 or between minimum_lot and maximum_order for authorized items; 0 for unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if any positive quantity of item `i` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected (i.e., both y[i_a] and y[i_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - `unit_benefit_cents` (per unit, from item tables)
        - `item_fee_cents` (fixed per item if any units selected, from item tables)
        - `activation_fee_cents` (fixed per category if any item in category selected, from category table)
        - `bonus_cents` (per bundle, from bundle table)
    -   Constraint coefficients:
        - `usage` (amount per item per resource, from usage tables)
        - `capacity_ledger` (sum of opening and reservation per resource, from capacity_ledger table)
        - `minimum_lot`, `maximum_order` (per item, from item tables)
        - `minimum_quantity`, `maximum_quantity` (per category, from category table)
        - `authorized` (per item, from item tables)
        - Incompatibility and prerequisite pairs (from respective tables)
6.  **Formulate Objective:** Maximize total net benefit in cents:
        - Sum over items: (unit_benefit_cents[i] * x[i]) 
        - Minus sum over items: (item_fee_cents[i] * y[i]) 
        - Minus sum over categories: (activation_fee_cents[c] * z[c])
        - Plus sum over bundles: (bonus_cents[b] * w[b])
7.  **Formulate Constraints:**
    -   Constraint 1 (Item Quantity Bounds): For each item `i`, x[i] = 0 if not authorized; else x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   Constraint 2 (Item Activation): For each item `i`, y[i] = 1 if x[i] > 0, else y[i] = 0.
    -   Constraint 3 (Resource/Area Capacity): For each resource/area `r`, sum over items assigned to `r` of (usage amount per unit * x[i]) ≤ total available capacity for `r` (sum of opening and reservation in capacity_ledger).
    -   Constraint 4 (Category Quantity Bounds): For each category `c`, sum over items in `c` of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   Constraint 5 (Category Activation): For each category `c`, z[c] = 1 if any x[i] > 0 for items in `c`, else z[c] = 0.
    -   Constraint 6 (Incompatibility): For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Constraint 7 (Prerequisite): For each requires pair (i, j), y[i] ≤ y[j].
    -   Constraint 8 (Bundle Bonus): For each bundle (i_a, i_b), w[b] = 1 if y[i_a] = 1 and y[i_b] = 1, else w[b] = 0.
[Abstract Model Plan END]