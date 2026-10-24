[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible items) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, item authorization, minimum/maximum order sizes, incompatibility and prerequisite (requires) logic, and bundle bonuses. All data is to be used as provided in the tables, with no expansion beyond the explicitly listed items.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee), generalized assignment, and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All item_refs from the union of item tables (authorized/unauthorized as specified).
    - Categories (`g`): All categories from the category table.
    - Resources (`r`): All resources from the capacity_ledger and usage tables.
    - Bundles (`(i,j)`): All item pairs from the bundle table.
    - Incompatible pairs (`(i,j)`): All item pairs from the incompatible table.
    - Requires pairs (`(i,j)`): All item-prerequisite pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (0 if not selected). Type: GRB.INTEGER, with bounds [0, maximum_order[i]] and, if authorized, [minimum_lot[i], maximum_order[i]] when selected.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is selected (i.e., sum_{i in g} y[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[i,j]` = 1 if both items `i` and `j` in a bundle are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of all amount_cents for each item_ref in the benefit table.
    -   Per-item activation fee: activation_fee_cents from the item_fee table.
    -   Per-category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from the category table.
    -   Item authorization, minimum_lot, maximum_order, and category: from the item tables (authorized == 1 means eligible).
    -   Resource usage per item: amount from the usage table, by item_ref and resource.
    -   Resource capacity: sum of amount for each resource in the capacity_ledger table.
    -   Bundle bonuses: bonus_cents from the bundle table for each (item_a, item_b) pair.
    -   Incompatibility: item pairs from the incompatible table.
    -   Requires: item-prerequisite pairs from the requires table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
        - Sum over all items: (per-item benefit) * x[i]
        - Minus: sum of item activation_fee_cents for each item with y[i] = 1
        - Minus: sum of category activation_fee_cents for each category with z[g] = 1
        - Plus: sum of bundle bonus_cents for each bundle where both items are selected (b[i,j] = 1)
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item, x[i] = 0 if authorized[i] == 0; otherwise, minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i], with y[i] ∈ {0,1}.
    -   **Resource Limits:** For each resource r, sum over all items of (usage amount of r per unit of i) * x[i] ≤ total available amount for r from capacity_ledger.
    -   **Category Quantity Bounds:** For each category g, sum over all items in g of x[i] ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    -   **Category Activation:** For each category g, z[g] = 1 if any y[i] in g is 1; z[g] = 0 otherwise (z[g] ≥ y[i] for all i in g; z[g] ≤ sum_{i in g} y[i]).
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Requires:** For each requires pair (i,j), y[i] ≤ y[j] (i requires j).
    -   **Bundle Bonuses:** For each bundle (i,j), b[i,j] ≤ y[i], b[i,j] ≤ y[j], b[i,j] ≥ y[i] + y[j] - 1.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; x[i] = 0 for unauthorized items; y[i], z[g], b[i,j] ∈ {0,1}.
[Abstract Model Plan END]