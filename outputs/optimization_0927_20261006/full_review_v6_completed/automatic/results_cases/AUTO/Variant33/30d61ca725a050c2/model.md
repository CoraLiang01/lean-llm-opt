[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible items) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, item authorization, incompatibility and prerequisite logic, and bundle bonuses. All data is to be used as provided in the tables, with explicit handling of authorized items, integer lot sizes, and logical constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee), generalized assignment, and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref) — all items listed in the item tables (from both batch_01/export_07.csv and batch_02/export_08.csv).
    - Categories (category) — as defined in the category table.
    - Resources (resource) — as defined in the usage and capacity_ledger tables.
    - Bundles (pairs of items) — as defined in the bundle table.
    - Incompatible pairs (item_a, item_b) — as defined in the incompatible table.
    - Requires pairs (item_ref, prerequisite_ref) — as defined in the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (in units, must be zero or between minimum_lot and maximum_order if authorized; zero if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum of y[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[a,b]` = 1 if both items a and b in a bundle are selected (i.e., y[a]=1 and y[b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: sum of all amount_cents for each item_ref in the benefit table.
    -   Per-item activation fee: activation_fee_cents from the item_fee table.
    -   Per-category activation fee: activation_fee_cents from the category table.
    -   Per-item resource usage: amount from the usage table, by item_ref and resource.
    -   Resource capacities: sum of amount for each resource in the capacity_ledger table (sum all entries for each resource).
    -   Item authorization, minimum_lot, maximum_order, and category assignment: from the item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Category quantity bounds: minimum_quantity and maximum_quantity from the category table.
    -   Bundle bonuses: bonus_cents from the bundle table.
    -   Incompatibility: item_a, item_b pairs from the incompatible table.
    -   Requires: item_ref, prerequisite_ref pairs from the requires table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (per-item benefit) × x[i]
    -   Minus: sum of item activation fees for each item selected (i.e., y[i]=1)
    -   Minus: sum of category activation fees for each category with at least one item selected (i.e., z[c]=1)
    -   Plus: sum of bundle bonuses for each bundle where both items are selected (i.e., b[a,b]=1)
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item i:
        - If authorized[i]=0, enforce x[i]=0 and y[i]=0.
        - If authorized[i]=1, enforce x[i]=0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
        - Link y[i] to x[i]: x[i] ≥ minimum_lot[i] × y[i], x[i] ≤ maximum_order[i] × y[i], and x[i]=0 ⇒ y[i]=0.
    -   **Resource limits:** For each resource r:
        - Sum over all items: (resource usage per unit for item i and resource r) × x[i] ≤ total available capacity for resource r (sum of capacity_ledger amounts for r).
    -   **Category quantity bounds:** For each category c:
        - Sum over all items in category c: minimum_quantity[c] ≤ sum x[i] ≤ maximum_quantity[c] (unconditional, even if no item is selected).
        - Link z[c] to y[i]: for all i in c, y[i] ≤ z[c]; z[c] ≤ sum over i in c of y[i].
    -   **Item activation fees:** For each item i, deduct activation_fee_cents only if y[i]=1.
    -   **Category activation fees:** For each category c, deduct activation_fee_cents only if z[c]=1.
    -   **Bundle bonuses:** For each bundle (a,b), set b[a,b]=1 iff y[a]=1 and y[b]=1; b[a,b]=0 otherwise.
    -   **Incompatibility:** For each incompatible pair (a,b), enforce y[a] + y[b] ≤ 1.
    -   **Requires:** For each requires pair (i, prereq), enforce y[i] ≤ y[prereq].
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers), y[i], z[c], b[a,b] ∈ {0,1}.
[Abstract Model Plan END]