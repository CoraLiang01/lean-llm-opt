[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models to assign to storage areas at FC_EAST_HVAC, maximizing total net value (in USD cents). The plan must respect per-area volume limits, per-category aggregate quantity bounds, item and category activation fees, item and bundle bonuses, incompatibility and prerequisite requirements, and item-specific order limits. Only authorized options may be selected; unauthorized items must have zero quantity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee), generalized assignment, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All item_ref rows from the union of item tables (all rows in tables tagged 'item').
    - Categories (`C`): All category rows from the category table.
    - Resources/Areas (`R`): All resource rows from the capacity_ledger and usage tables (e.g., AREA_A, AREA_B, AREA_C).
    - Bundles (`B`): All bundle rows from the bundle table.
    - Incompatibility pairs (`INC`): All pairs from the incompatible table.
    - Prerequisite pairs (`REQ`): All pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (for each item_ref in I). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected (i.e., category is activated), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   For each item `i`:
        -   `unit_benefit_cents[i]` and `item_fee_cents[i]` from item tables.
        -   `authorized[i]`, `minimum_lot[i]`, `maximum_order[i]`, `category[i]`, `location_id[i]`.
    -   For each category `c`:
        -   `minimum_quantity[c]`, `maximum_quantity[c]`, `activation_fee_cents[c]` from category table.
    -   For each resource/area `r`:
        -   Total available capacity: sum of `amount` for resource `r` in capacity_ledger table.
        -   For each item `i`, `usage[i,r]` from usage tables (amount for item_ref=i, resource=r).
    -   For each bundle `b`:
        -   `item_a[b]`, `item_b[b]`, `bonus_cents[b]` from bundle table.
    -   Incompatibility and prerequisite pairs from respective tables.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) − (item_fee_cents[i] * y[i])
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c])
    -   Plus sum over all bundles: (bonus_cents[b] * w[b])
7.  **Formulate Constraints:**
    -   **Item Authorization:** For each item `i`, if authorized[i] = 0, enforce x[i] = 0 and y[i] = 0.
    -   **Item Quantity Bounds:** For each item `i`, minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; x[i] ≥ 0 integer; y[i] ∈ {0,1}.
    -   **Category Aggregate Bounds:** For each category `c`, sum_{i in c} x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c]; z[c] = 1 iff any y[i]=1 for i in c.
    -   **Category Activation:** For each category `c`, for all i in c: y[i] ≤ z[c]; z[c] ≤ sum_{i in c} y[i].
    -   **Resource/Area Capacity:** For each resource/area `r`, sum over all items assigned to r: sum_{i: location_id[i]=r} (usage[i,r] * x[i]) ≤ total available capacity for r (sum of capacity_ledger amounts for r).
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each requires pair (i,prereq), y[i] ≤ y[prereq].
    -   **Bundle Bonuses:** For each bundle b with items (i,j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] − 1.
    -   **Variable Domains:** All variables as above; all sums and constraints apply only to the selected (authorized) items and categories.
[Abstract Model Plan END]