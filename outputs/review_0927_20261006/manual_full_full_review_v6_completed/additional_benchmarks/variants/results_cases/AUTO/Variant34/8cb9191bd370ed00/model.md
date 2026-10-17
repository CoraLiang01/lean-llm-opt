[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each authorized product-and-section option (item_ref at location_id), maximizing net merchandising benefit. The plan must respect section capacities, per-item and per-category activation fees, minimum/maximum order sizes, resource usage limits, category-wide quantity bounds, incompatibility and prerequisite requirements, and bundle bonuses. Only authorized options may be selected; unauthorized options must have zero quantity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref-location_id options from the union of item tables (joined on item_ref, location_id).
    - Categories: All categories from the category table.
    - Resources: All resources from the capacity_ledger and usage tables.
    - Bundles: All bundle pairs from the bundle table.
    - Incompatibility pairs: All item_ref pairs from the incompatible table.
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs to display for item option i (item_ref at location_id). Type: GRB.INTEGER, domain: 0 or [minimum_lot, maximum_order] if authorized, 0 if unauthorized.
    -   `y[i]` = 1 if item option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (category is active), 0 otherwise. Type: GRB.BINARY.
    -   `b[p]` = 1 if both items in bundle pair p are selected (both y[i_a] = y[i_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: Sum of amount_cents for each item_ref from the benefit table.
    -   Item activation fee: activation_fee_cents from the item_fee table, per item_ref.
    -   Category activation fee: activation_fee_cents from the category table, per category.
    -   Bundle bonus: bonus_cents from the bundle table, per bundle pair.
    -   Section capacity: sum of amount (converted to ml) for each resource from capacity_ledger table (opening + reservation).
    -   Item resource usage: amount (converted to ml) from usage table, per item_ref and resource.
    -   Minimum/maximum order: minimum_lot, maximum_order from item tables, per item_ref-location_id.
    -   Authorization: authorized from item tables, per item_ref-location_id.
    -   Category bounds: minimum_quantity, maximum_quantity from category table, per category.
    -   Item category: category from item tables, per item_ref-location_id.
    -   Incompatibility: item_a, item_b from incompatible table.
    -   Prerequisites: item_ref, prerequisite_ref from requires table.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
        - Sum over all items: (per-unit benefit * x[i]) 
        - Minus sum over all selected items: item activation_fee_cents * y[i]
        - Minus sum over all active categories: category activation_fee_cents * z[c]
        - Plus sum over all active bundles: bonus_cents * b[p]
      All terms are in USD cents.
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item option i, x[i] = 0 if not authorized; if authorized, x[i] = 0 or minimum_lot ≤ x[i] ≤ maximum_order. Enforce y[i] = 1 iff x[i] > 0.
    -   **Section Capacity:** For each resource (section), sum over all items assigned to that resource of (item usage per pack in ml * x[i]) ≤ total available capacity in ml (sum of capacity_ledger entries for that resource).
    -   **Category Quantity Bounds:** For each category c, sum over all items in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   **Category Activation:** For each category c, z[c] = 1 iff any y[i] = 1 for items in c; z[c] = 0 otherwise.
    -   **Item Activation Fee:** For each item i, item activation fee is charged iff y[i] = 1 (enforced via objective).
    -   **Category Activation Fee:** For each category c, category activation fee is charged iff z[c] = 1 (enforced via objective).
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq), y[i] ≤ y[prereq].
    -   **Bundle Bonus:** For each bundle pair (i, j), b[p] = 1 iff y[i] = y[j] = 1; b[p] ≤ y[i], b[p] ≤ y[j], b[p] ≥ y[i] + y[j] - 1.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integers) if authorized, x[i] = 0 if not authorized; y[i], z[c], b[p] ∈ {0,1}.
[Abstract Model Plan END]