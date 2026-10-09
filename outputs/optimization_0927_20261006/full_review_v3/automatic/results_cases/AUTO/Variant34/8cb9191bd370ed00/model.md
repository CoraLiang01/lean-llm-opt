[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of packs to display for each authorized product-and-section option (item_ref), maximizing total net merchandising benefit for MARKET_SQUARE. The plan must respect section capacities, per-option and per-category limits, fixed and bundle fees, incompatibility and prerequisite rules, and resource usage, using only the supplied table rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource allocation, and logical (linking) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all item_ref values from the supplied item tables (union of batch_01/export_07.csv and batch_02/export_08.csv).
    - Sections/resources: all unique location_id/resource values from the relevant tables.
    - Categories: all category values from the category table.
    - Bundles: all (item_a, item_b) pairs from the bundle table.
    - Incompatibility and prerequisite pairs: from the incompatible and requires tables.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs of item_ref i to display (integer, 0 if unauthorized or not chosen; between minimum_lot and maximum_order if chosen). Type: GRB.INTEGER.
    -   `z[i]` = 1 if item_ref i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = 1 if any item in category c is selected (i.e., sum over i in c of z[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., z[item_a] = z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item_ref: sum of amount_cents from all benefit rows for that item_ref (batch_01/export_01.csv).
    -   Item activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv).
    -   Bundle bonus: bonus_cents from bundle table (batch_02/export_02.csv).
    -   Section/resource usage per item: amount (converted to ml if needed) from usage tables (batch_06/export_12.csv and batch_01/export_13.csv), mapped by item_ref and resource.
    -   Section/resource total available: sum of amount from capacity_ledger table (batch_03/export_03.csv), per resource.
    -   Category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table (batch_04/export_04.csv).
    -   Authorization, minimum_lot, maximum_order, category, location_id: from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Incompatibility pairs: from incompatible table (batch_06/export_06.csv).
    -   Prerequisite pairs: from requires table (batch_05/export_11.csv).
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all item_ref of (per-unit benefit × x[i])
    -   Minus sum over all item_ref of (item activation fee × z[i])
    -   Minus sum over all categories of (category activation fee × w[c])
    -   Plus sum over all bundles of (bundle bonus × b[bundle])
    -   (All terms in USD cents; no conversion needed.)
7.  **Formulate Constraints:**
    -   **Authorization and Bounds:** For each item_ref, x[i] = 0 if not authorized; otherwise, minimum_lot[i] × z[i] ≤ x[i] ≤ maximum_order[i] × z[i], with z[i] ∈ {0,1}.
    -   **Section/Resource Capacity:** For each resource (section), sum over all item_ref assigned to that resource of (usage per pack × x[i]) ≤ total available capacity (in ml).
    -   **Category Quantity Bounds:** For each category, sum over all item_ref in that category of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   **Category Activation:** For each category, w[c] = 1 if any z[i] = 1 for i in c; w[c] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   **Prerequisite:** For each requires pair (i, j), z[i] ≤ z[j].
    -   **Bundle Bonus Activation:** For each bundle (item_a, item_b), b[bundle] ≤ z[item_a], b[bundle] ≤ z[item_b], and b[bundle] ≥ z[item_a] + z[item_b] - 1.
    -   **Integrality:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), z[i], w[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]