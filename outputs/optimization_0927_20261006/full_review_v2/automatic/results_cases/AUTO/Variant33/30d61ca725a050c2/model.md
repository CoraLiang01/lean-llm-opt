[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible items) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item and category activation fees, incompatibility and prerequisite logic, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ Items): All item_ref values from the union of both item tables (export_07.csv and export_08.csv), filtered to authorized=1.
    - Categories (g ∈ Categories): All category values from the category table (export_04.csv).
    - Resources (r ∈ Resources): All resource values from the usage and capacity_ledger tables.
    - Bundles (b ∈ Bundles): All rows in the bundle table (export_02.csv).
    - Incompatibility pairs (p ∈ Incompatibles): All (item_a, item_b) pairs from the incompatible table (export_06.csv).
    - Prerequisite pairs (q ∈ Requires): All (item_ref, prerequisite_ref) pairs from the requires table (export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected). Type: GRB.INTEGER, with bounds [minimum_lot[i], maximum_order[i]] if selected, 0 otherwise.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is selected (category is "used"), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of amount_cents for each item_ref from the benefit table (export_01.csv).
    -   Per-item activation fee: activation_fee_cents from item_fee table (export_09.csv).
    -   Per-category activation fee: activation_fee_cents from category table (export_04.csv).
    -   Per-item resource usage: amount from usage tables (export_12.csv and export_13.csv), by item_ref and resource.
    -   Resource capacities: sum of amount for each resource from capacity_ledger table (export_03.csv).
    -   Item-category mapping: from item tables (export_07.csv and export_08.csv).
    -   Item authorization, minimum_lot, maximum_order: from item tables (export_07.csv and export_08.csv).
    -   Bundle bonuses: bonus_cents from bundle table (export_02.csv).
    -   Incompatibility and prerequisite pairs: from incompatible (export_06.csv) and requires (export_11.csv).
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table (export_04.csv).
6.  **Formulate Objective:** Maximize total net benefit in cents:
        - Sum over all items: (per-unit benefit[i] * x[i]) 
        - Minus sum over all selected items: item_fee[i] * y[i]
        - Minus sum over all used categories: category activation_fee_cents[g] * z[g]
        - Plus sum over all activated bundles: bonus_cents[b] * w[b]
7.  **Formulate Constraints:**
    -   Resource Limits: For each resource r, sum over all items of (resource usage per unit[i, r] * x[i]) ≤ total available capacity[r] (from capacity_ledger).
    -   Item Selection Bounds: For each item i, x[i] = 0 if not authorized; otherwise, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
    -   Item Activation Linking: For each item i, y[i] = 1 if x[i] > 0, y[i] = 0 if x[i] = 0.
    -   Category Quantity Bounds: For each category g, sum over items in g of x[i] ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    -   Category Activation Linking: For each category g, z[g] = 1 if any x[i] > 0 for i in g; z[g] = 0 otherwise.
    -   Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Prerequisite: For each (i, prereq), y[i] ≤ y[prereq].
    -   Bundle Activation: For each bundle b = (i, j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1; w[b] = 1 iff both y[i] = y[j] = 1.
[Abstract Model Plan END]