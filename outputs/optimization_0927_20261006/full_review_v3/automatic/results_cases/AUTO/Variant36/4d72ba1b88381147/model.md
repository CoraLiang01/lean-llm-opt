[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer quantities of authorized air-conditioner models to assign to each storage area at FC_EAST_HVAC, maximizing net value (total benefit plus bonuses minus all fixed and activation fees), subject to per-area volume limits, item and category quantity bounds, incompatibility and prerequisite rules, and bundle bonuses. Each area’s capacity is independent, and only authorized options may be selected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, assignment, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All item_ref rows from the supplied item tables (only those with authorized=1).
    - Categories (`C`): All category rows from the category table.
    - Resources/Areas (`R`): All resource/location_id values from the capacity_ledger and usage tables.
    - Bundles (`B`): All bundle rows.
    - Incompatibility pairs (`INC`): All incompatible item pairs.
    - Prerequisite pairs (`REQ`): All requires pairs.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (0 if not selected). Type: GRB.INTEGER. Domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; 0 for unauthorized.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected (i.e., sum_{i in c} y[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from item tables): per-unit benefit for item `i`.
        -   `item_fee_cents[i]` (from item tables): fixed fee if any of item `i` is selected.
        -   `activation_fee_cents[c]` (from category table): fixed fee if any item in category `c` is selected.
        -   `bonus_cents[b]` (from bundle table): bonus if both items in bundle `b` are selected.
    -   Constraint coefficients:
        -   `usage[i, r]` (from usage tables): amount of resource `r` used per unit of item `i`.
        -   `capacity[r]` (from capacity_ledger): total available for resource `r` (sum of opening and reservation entries).
        -   `minimum_lot[i]`, `maximum_order[i]` (from item tables): unconditional lower and upper bounds for x[i].
        -   `minimum_quantity[c]`, `maximum_quantity[c]` (from category table): unconditional lower and upper bounds for total quantity in category `c`.
    -   Logical constraints:
        -   Incompatibility pairs: from incompatible table.
        -   Prerequisite pairs: from requires table.
        -   Bundle pairs: from bundle table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
        sum_{i in I} (unit_benefit_cents[i] * x[i] - item_fee_cents[i] * y[i])
      + sum_{b in B} (bonus_cents[b] * b[b])
      - sum_{c in C} (activation_fee_cents[c] * z[c])
7.  **Formulate Constraints:**
    -   Resource/Area Capacity: For each resource/area `r`, sum over all items assigned to `r` of (usage[i, r] * x[i]) ≤ capacity[r].
    -   Item Selection and Bounds: For each item `i`, x[i] = 0 if unauthorized; otherwise, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]. Enforce y[i] = 1 if x[i] > 0, y[i] = 0 if x[i] = 0.
    -   Category Quantity Bounds: For each category `c`, sum_{i in c} x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c] (unconditional).
    -   Category Activation: For each category `c`, z[c] = 1 if any y[i] in c is 1, else 0.
    -   Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Prerequisite: For each requires pair (i, j), y[i] ≤ y[j].
    -   Bundle Bonuses: For each bundle (i, j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1.
    -   Integrality: All x[i] integer, all y[i], z[c], b[bundle] binary.
[Abstract Model Plan END]