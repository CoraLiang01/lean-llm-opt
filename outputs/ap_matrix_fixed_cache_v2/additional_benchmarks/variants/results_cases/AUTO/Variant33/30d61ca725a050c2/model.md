[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of authorized, indivisible development modules (items), maximizing total net benefit (sum of per-unit benefits, minus item and category activation fees, plus bundle bonuses), subject to resource (storage, labor, energy) limits, category quantity bounds, incompatibility and prerequisite logic, and bundle bonus conditions. Each item can be chosen in integer multiples within its allowed lot/order range or not at all; unauthorized items must be excluded.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i): All item_ref values from the union of item tables (authorized subset only).
    - Categories (g): All category values from the category table.
    - Resources (r): All resource values from the capacity_ledger and usage tables.
    - Bundles (b): All bundle rows (pairs of items) from the bundle table.
    - Incompatible pairs (p): All incompatible item pairs from the incompatible table.
    - Requires pairs (q): All prerequisite pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 or in [minimum_lot, maximum_order] if authorized; 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is selected (category activated), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected (bundle bonus applies), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of amount_cents for each item_ref in the benefit table (all components for that item).
    -   Item activation fee: activation_fee_cents from item_fee table, by item_ref.
    -   Category activation fee: activation_fee_cents from category table, by category.
    -   Bundle bonus: bonus_cents from bundle table, by bundle (item_a, item_b).
    -   Resource usage per item: amount from usage table, by (item_ref, resource).
    -   Resource capacity: sum of amount for each resource in capacity_ledger table (sum all entries for each resource).
    -   Category bounds: minimum_quantity and maximum_quantity from category table, by category.
    -   Item eligibility and bounds: authorized, minimum_lot, maximum_order from item tables, by item_ref.
    -   Incompatibility: item_a, item_b pairs from incompatible table.
    -   Prerequisite: item_ref, prerequisite_ref pairs from requires table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over items: (per-unit benefit[i] * x[i]) 
    -   Minus sum over items: (item activation_fee[i] * y[i])
    -   Minus sum over categories: (category activation_fee[g] * z[g])
    -   Plus sum over bundles: (bundle bonus[b] * w[b])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item i:
        - If authorized[i] = 1: x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i] = 1 iff x[i] > 0.
        - If authorized[i] = 0: x[i] = 0, y[i] = 0.
        - Enforce: y[i] ≤ x[i]/minimum_lot[i] ≤ y[i]*maximum_order[i]/minimum_lot[i] (or equivalent logic).
    -   **Resource limits:** For each resource r:
        - sum over items i of (usage[i, r] * x[i]) ≤ total capacity[r] (sum of capacity_ledger amounts for r).
    -   **Category quantity bounds:** For each category g:
        - sum over items i in category g of x[i] ≥ minimum_quantity[g]
        - sum over items i in category g of x[i] ≤ maximum_quantity[g]
        - z[g] = 1 iff any y[i] = 1 for i in g; z[g] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1 (cannot select both).
    -   **Prerequisite:** For each requires pair (i, j):
        - y[i] ≤ y[j] (if i is selected, j must also be selected).
    -   **Bundle bonuses:** For each bundle (i, j):
        - w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1 (w[b] = 1 iff both y[i] = y[j] = 1).
    -   **Variable domains:** x[i] integer, y[i], z[g], w[b] binary.
[Abstract Model Plan END]