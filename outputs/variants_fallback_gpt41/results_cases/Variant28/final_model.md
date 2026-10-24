[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using the supplied tables as the complete set of options.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (from all rows in the item table, filtered to authorized=1)
    - Categories (from all rows in the category table)
    - Resources (from all rows in the resource and usage tables)
    - Incompatible pairs (from all rows in the incompatible table)
    - Prerequisite pairs (from all rows in the requires table)
    - Bundles (from all rows in the bundle table)
4.  **Define Decision Variables:**
    -   `q[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items.
    -   `z[i]` = 1 if item `i` is ordered (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[c]` = 1 if any item in category `c` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle `bundle` are ordered, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from item table): per-unit benefit for item `i`.
        -   `item_fee_cents[i]` (from item table): fixed fee if any of item `i` is ordered.
        -   `activation_fee_cents[c]` (from category table): fixed fee if any item in category `c` is ordered.
        -   `bonus_cents[bundle]` (from bundle table): bonus if both items in bundle are ordered.
    -   Constraint coefficients:
        -   `amount[i,r]` (from usage table): per-unit usage of resource `r` by item `i`, with unit conversion as needed.
    -   Constraint RHS:
        -   Resource capacities: sum of `amount` in capacity_ledger table for each resource, after converting all units to a common base (ml, wh, minute).
        -   Category minimum_quantity and maximum_quantity (from category table).
        -   Incompatible pairs (from incompatible table).
        -   Prerequisite pairs (from requires table).
        -   Minimum and maximum order per item (from item table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (unit_benefit_cents[i] * q[i]) 
    -   Minus sum over items: (item_fee_cents[i] * z[i])
    -   Minus sum over categories: (activation_fee_cents[c] * y[c])
    -   Plus sum over bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Item selection and quantity bounds:** For each authorized item `i`, enforce q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i], with z[i] = 1 iff q[i] > 0.
    -   **Category activation:** For each category `c`, y[c] = 1 iff any item in category `c` is ordered (z[i] for i in c).
    -   **Category quantity bounds:** For each category `c`, sum over i in c of q[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   **Resource capacity:** For each resource `r`, sum over items i of (amount[i,r] * q[i]) ≤ total available capacity for r (sum of capacity_ledger entries for r, after unit conversion).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq), z[i] ≤ z[prereq].
    -   **Bundle bonuses:** For each bundle (i, j), b[bundle] = 1 iff z[i] = 1 and z[j] = 1; b[bundle] ≤ z[i], b[bundle] ≤ z[j], b[bundle] ≥ z[i] + z[j] - 1. Only authorized items can trigger a bundle.
    -   **Variable domains:** q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), z[i], y[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]