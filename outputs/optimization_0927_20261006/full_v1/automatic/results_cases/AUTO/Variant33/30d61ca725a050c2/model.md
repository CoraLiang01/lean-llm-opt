[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefits plus bonuses minus setup and activation fees), subject to resource (storage, labor, energy) limits, category quantity bounds, item authorization, incompatibility and prerequisite rules, and bundle bonuses. All data is to be used as provided in the tables, with each item’s benefit as the sum of its benefit components, and all constraints and fees applied as described.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (setup/activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ Items): All item_ref values from the union of item tables (authorized items only).
    - Categories (g ∈ Categories): All category values from the category table.
    - Resources (r ∈ Resources): All resource values from the capacity_ledger and usage tables.
    - Bundles (b ∈ Bundles): All rows in the bundle table, each with a pair of items.
    - Incompatibility pairs (p ∈ Incompatibles): All item_a, item_b pairs from the incompatible table.
    - Prerequisite pairs (q ∈ Requires): All item_ref, prerequisite_ref pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected). Type: GRB.INTEGER, with bounds [0 or minimum_lot[i], maximum_order[i]] as authorized.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is selected (category is used), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of amount_cents for each item_ref in the benefit table.
    -   Item setup fee: activation_fee_cents from the item_fee table, per item_ref.
    -   Category activation fee: activation_fee_cents from the category table, per category.
    -   Resource usage: amount from usage tables, per item_ref and resource.
    -   Resource capacity: sum of amount for each resource in capacity_ledger table.
    -   Item authorization, minimum_lot, maximum_order, and category: from item tables (authorized == 1).
    -   Category quantity bounds: minimum_quantity and maximum_quantity from category table.
    -   Incompatibility: item_a, item_b pairs from incompatible table.
    -   Prerequisite: item_ref, prerequisite_ref pairs from requires table.
    -   Bundle bonus: bonus_cents from bundle table, per bundle.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over items: (per-item benefit) × x[i]
    -   Minus sum over items: item setup fee × y[i] (charged once per item if selected)
    -   Minus sum over categories: category activation fee × z[g] (charged once per category if any item in g is selected)
    -   Plus sum over bundles: bundle bonus × w[b] (awarded if both items in bundle are selected)
7.  **Formulate Constraints:**
    -   Resource Limits: For each resource r, sum over items of (resource usage per unit × x[i]) ≤ total available capacity for r (sum of capacity_ledger amounts for r).
    -   Item Authorization and Bounds: For each item i, x[i] = 0 if authorized[i] == 0; otherwise, minimum_lot[i] ≤ x[i] ≤ maximum_order[i] or x[i] = 0.
    -   Linking x and y: For each item i, y[i] = 1 if x[i] ≥ minimum_lot[i], y[i] = 0 if x[i] = 0; enforce x[i] ≤ maximum_order[i] × y[i].
    -   Category Quantity Bounds: For each category g, sum over items in g of x[i] ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    -   Category Activation: For each category g, z[g] = 1 if any x[i] > 0 for i in g; z[g] = 0 otherwise.
    -   Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Prerequisite: For each (i, prereq) in requires, y[i] ≤ y[prereq].
    -   Bundle Bonus Linking: For each bundle b = (i, j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1.
[Abstract Model Plan END]