[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using only the supplied item rows and respecting all specified limits and conditions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All item rows from the item table (export_06.csv) with authorized=1.
    - Categories (`C`): All category rows from the category table (export_03.csv).
    - Resources (`R`): All resources from the usage and capacity_ledger tables.
    - Incompatible pairs (`P`): All pairs from the incompatible table (export_05.csv).
    - Prerequisite pairs (`Q`): All pairs from the requires table (export_08.csv).
    - Bundles (`B`): All bundle rows from the bundle table (export_01.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for each authorized item.
    -   `y[i]` = 1 if item `i` is ordered (i.e., x[i] ≥ minimum_lot[i]), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are ordered (i.e., x[item_a[b]] > 0 and x[item_b[b]] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from item table): per-unit benefit for item `i`.
        -   `item_fee_cents[i]` (from item table): fixed fee if any of item `i` is ordered.
        -   `activation_fee_cents[c]` (from category table): fixed fee if any item in category `c` is ordered.
        -   `bonus_cents[b]` (from bundle table): bonus if both items in bundle `b` are ordered.
    -   Constraint coefficients:
        -   `amount[i, r]` and `unit[i, r]` (from usage table): per-unit resource usage for item `i` and resource `r`.
        -   `amount[r]` and `unit[r]` (from capacity_ledger): total available capacity for resource `r` (sum of opening and reservation entries).
    -   Constraint RHS:
        -   `minimum_quantity[c]`, `maximum_quantity[c]` (from category table): total quantity bounds for each category.
        -   `minimum_lot[i]`, `maximum_order[i]` (from item table): per-item order bounds.
        -   Incompatible pairs and prerequisites: as listed in their respective tables.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        sum over items i of (unit_benefit_cents[i] * x[i] - item_fee_cents[i] * y[i])
      plus sum over bundles b of (bonus_cents[b] * w[b])
      minus sum over categories c of (activation_fee_cents[c] * z[c]).
7.  **Formulate Constraints:**
    -   Resource Capacity: For each resource r, sum over items i of (x[i] * amount[i, r] * unit_conversion[i, r]) ≤ total available capacity for r (converted to base units: ml, wh, minute).
    -   Item Order Bounds: For each item i, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   Item Activation: For each item i, y[i] = 1 if x[i] ≥ minimum_lot[i], y[i] = 0 if x[i] = 0; enforce with x[i] ≥ minimum_lot[i] * y[i] and x[i] ≤ maximum_order[i] * y[i].
    -   Category Quantity Bounds: For each category c, sum over items i in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   Category Activation: For each category c, z[c] = 1 if any x[i] > 0 for i in c, 0 otherwise; enforce with z[c] ≥ y[i] for all i in c.
    -   Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Prerequisite: For each (i, j) where i requires j, y[i] ≤ y[j].
    -   Bundle Bonus: For each bundle b with items (i, j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1; only bundles where both items are authorized can be triggered.
[Abstract Model Plan END]