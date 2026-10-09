[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of authorized, indivisible development modules (items) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite logic, and bundle bonuses. Each item can be chosen in integer multiples within its allowed lot/order range or not at all; unauthorized items must be excluded.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee), generalized assignment, and logical (incompatibility, prerequisite, bundle) constraints.
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
    -   `z[g]` = 1 if any item in category g is selected (category is active), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of amount_cents for each item_ref in the benefit table (all components for that item).
    -   Per-item activation fee: activation_fee_cents from the item_fee table, by item_ref.
    -   Per-category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from the category table, by category.
    -   Item-category mapping and eligibility: category, authorized, minimum_lot, maximum_order from item tables, by item_ref.
    -   Resource usage per item: amount from usage table, by (item_ref, resource).
    -   Resource capacity: sum of amount for each resource in capacity_ledger table (sum all entries for each resource).
    -   Incompatibility: item_a, item_b pairs from incompatible table.
    -   Prerequisite: item_ref, prerequisite_ref pairs from requires table.
    -   Bundle bonuses: item_a, item_b, bonus_cents from bundle table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over items: (per-item benefit) × x[i]
    -   Minus: sum of item activation_fee_cents for each item with y[i]=1
    -   Minus: sum of category activation_fee_cents for each category with z[g]=1
    -   Plus: sum of bundle bonus_cents for each bundle with w[b]=1
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item i:
        - If authorized: x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (enforced via y[i]: minimum_lot[i] × y[i] ≤ x[i] ≤ maximum_order[i] × y[i]; x[i]=0 iff y[i]=0).
        - If unauthorized: x[i]=0, y[i]=0.
    -   **Category quantity bounds:** For each category g: sum of x[i] over items in g ≥ minimum_quantity[g] and ≤ maximum_quantity[g] (unconditional, applies even if z[g]=0).
    -   **Category activation:** For each category g: z[g]=1 iff any y[i]=1 for i in g (z[g] ≥ y[i] for all i in g; z[g] ≤ sum of y[i] over i in g).
    -   **Resource limits:** For each resource r: sum over items of (usage amount per unit × x[i]) ≤ total available capacity for r (sum of capacity_ledger amounts for r).
    -   **Item activation:** For each item i: y[i]=1 iff x[i]>0 (enforced via bounds above).
    -   **Incompatibility:** For each incompatible pair (i,j): y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each requires pair (i,pr): y[i] ≤ y[pr].
    -   **Bundle bonuses:** For each bundle (i,j): w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1 (w[b]=1 iff both y[i]=y[j]=1).
[Abstract Model Plan END]