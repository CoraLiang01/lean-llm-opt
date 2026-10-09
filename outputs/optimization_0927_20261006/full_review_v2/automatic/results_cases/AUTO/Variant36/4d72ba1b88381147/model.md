[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer quantities of authorized air-conditioner models to assign to storage areas at FC_EAST_HVAC, maximizing net value (benefit minus all fixed and variable fees, plus any applicable bundle bonuses), subject to area volume limits, category quantity bounds, incompatibility and prerequisite rules, and bundle bonuses, using only the supplied item rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, assignment, and logical (incompatibility/prerequisite) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): Each authorized item_ref from the supplied item tables.
    - Categories (`C`): Each category from the category table.
    - Resources/Areas (`R`): Each resource/location_id from the capacity_ledger and usage tables.
    - Bundles (`B`): Each bundle pair from the bundle table.
    - Incompatibility pairs and prerequisite pairs from their respective tables.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to assign (must be 0 or between minimum_lot and maximum_order for authorized items; 0 for unauthorized). Type: GRB.INTEGER.
    -   `z[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[c]` = 1 if any item in category `c` is selected (i.e., sum over i in c of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected (i.e., z[item_a] = z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (per unit, from item tables).
        -   `item_fee_cents` (fixed per item if any units selected, from item tables).
        -   `activation_fee_cents` (fixed per category if any item in category selected, from category table).
        -   `bonus_cents` (per bundle, from bundle table).
    -   Constraint coefficients:
        -   `amount` (resource usage per item, from usage tables).
        -   `amount` (resource capacity per area, from capacity_ledger table; sum opening and reservation for each resource).
        -   `minimum_lot`, `maximum_order` (per item, from item tables).
        -   `minimum_quantity`, `maximum_quantity` (per category, from category table).
        -   Incompatibility and prerequisite pairs (from respective tables).
    -   Constraint RHS:
        -   Resource capacities (from summed capacity_ledger entries per resource).
        -   Category quantity bounds (from category table).
6.  **Formulate Objective:** Maximize total net benefit in cents:
        -   Sum over items: (unit_benefit_cents[i] * x[i]) 
        -   Minus sum over items: (item_fee_cents[i] * z[i])
        -   Minus sum over categories: (activation_fee_cents[c] * y[c])
        -   Plus sum over bundles: (bonus_cents[b] * w[b])
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item, x[i] = 0 if not authorized; otherwise, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item Selection Indicator:** For each item, z[i] = 1 if x[i] > 0, z[i] = 0 if x[i] = 0 (enforced via standard big-M or indicator constraints).
    -   **Category Quantity Bounds:** For each category, sum over items in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c] (applies unconditionally, even if y[c]=0).
    -   **Category Activation Indicator:** For each category, y[c] = 1 if any z[i] = 1 for i in c; y[c] = 0 otherwise.
    -   **Resource/Area Capacity:** For each resource/area, sum over items assigned to that area of (usage amount per unit * x[i]) ≤ total available capacity (sum of opening and reservation entries for that resource).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   **Prerequisite:** For each requires pair (i, j), z[i] ≤ z[j] (i.e., if i is selected, j must also be selected).
    -   **Bundle Bonus:** For each bundle (item_a, item_b), w[b] = 1 if z[item_a] = z[item_b] = 1; w[b] = 0 otherwise.
[Abstract Model Plan END]