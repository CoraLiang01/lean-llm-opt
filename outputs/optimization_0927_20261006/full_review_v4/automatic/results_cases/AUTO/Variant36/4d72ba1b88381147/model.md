[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to assign to storage areas at FC_EAST_HVAC, maximizing total net benefit (in USD cents). The plan must respect per-area volume limits, item and category quantity bounds, fixed and activation fees, incompatibility and prerequisite requirements, and bundle bonuses, using only the supplied item rows and categories.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, group activation, logical, and resource constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All item rows from the union of item tables (from both batch_06/export_06.csv and batch_01/export_07.csv).
    - Categories (`C`): All category rows from the category table (batch_03/export_03.csv).
    - Resources/Areas (`R`): All resources from the capacity_ledger and usage tables (e.g., AREA_A, AREA_B, AREA_C).
    - Bundles (`B`): All bundle rows (batch_01/export_01.csv).
    - Incompatibility pairs (`Inc`): All incompatible item pairs (batch_05/export_05.csv).
    - Prerequisite pairs (`Req`): All requires pairs (batch_03/export_09.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` to assign (0 if not selected, or between minimum_lot and maximum_order if selected). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected (i.e., category is activated), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   For each item `i`:
        -   `unit_benefit_cents[i]` and `item_fee_cents[i]` (from item tables).
        -   `authorized[i]`, `minimum_lot[i]`, `maximum_order[i]`, `category[i]`, `location_id[i]`.
    -   For each category `c`:
        -   `minimum_quantity[c]`, `maximum_quantity[c]`, `activation_fee_cents[c]` (from category table).
    -   For each resource/area `r`:
        -   Capacity: sum of `amount` for resource `r` in capacity_ledger table.
        -   For each item `i`, `usage[i,r]`: amount from usage tables for item `i` and resource `r` (0 if missing).
    -   For each bundle `b`:
        -   `item_a[b]`, `item_b[b]`, `bonus_cents[b]` (from bundle table).
    -   Incompatibility and prerequisite pairs as sets of item pairs.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) − (item_fee_cents[i] * y[i])
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c])
    -   Plus sum over all bundles: (bonus_cents[b] * w[b])
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item `i`, if authorized[i] = 0, enforce x[i] = 0 and y[i] = 0. If authorized[i] = 1, enforce:
        -   x[i] = 0 if y[i] = 0; minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; x[i] integer; y[i] binary.
    -   **Category Quantity Bounds:** For each category `c`, sum of x[i] over items in `c` must satisfy:
        -   minimum_quantity[c] ≤ sum_{i in c} x[i] ≤ maximum_quantity[c]
    -   **Category Activation:** For each category `c`, z[c] = 1 if any y[i] = 1 for i in c, else 0. Enforce y[i] ≤ z[c] for all i in c, and z[c] ≤ sum_{i in c} y[i].
    -   **Resource/Area Capacity:** For each resource/area `r`, sum over all items of (usage[i,r] * x[i]) ≤ total capacity for r (sum of capacity_ledger amounts for r).
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each requires pair (i, j), y[i] ≤ y[j].
    -   **Bundle Bonuses:** For each bundle b = (i, j), w[b] = 1 iff y[i] = 1 and y[j] = 1; enforce w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] − 1.
    -   **Variable Domains:** All x[i] integer, y[i], z[c], w[b] binary.
[Abstract Model Plan END]