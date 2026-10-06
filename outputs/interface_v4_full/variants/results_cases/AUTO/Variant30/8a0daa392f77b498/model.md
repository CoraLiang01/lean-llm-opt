[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The solution must respect inventory, resource, category, compatibility, dependency, and authorization constraints, and must account for all fixed and per-unit costs/benefits, including bundle bonuses and activation fees. Data selection must strictly follow the "latest effective, highest revision, not DELETE" rule for each (tenant, table, record_id) before any joins or aggregations.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, compatibility, and dependency constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All item_ref values for authorized options in business unit NORTH, after applying the selection rule.
    - Categories (`g`): All categories associated with items in NORTH, after selection.
    - Resources (`r`): All resources (e.g., labor, space, power) relevant to item usage and capacity in NORTH.
    - Bundles (`b`): All bundle pairs (item_a, item_b) in NORTH, after selection.
    - Incompatible pairs (`(i,j)`): All incompatible item pairs in NORTH, after selection.
    - Requires pairs (`(i,k)`): All requires (dependency) pairs in NORTH, after selection.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity to order of item/option `i` (must be 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item/option `i` is ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered (category is "used"), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of `amount_cents` from benefit tables (tables: benefit), grouped by item_ref, after selection.
    -   Per-item fixed fee: `activation_fee_cents` from item_fee tables, by item_ref, after selection.
    -   Bundle bonus: `bonus_cents` from bundle tables, by (item_a, item_b), after selection.
    -   Category activation fee: `activation_fee_cents` from category tables, by category, after selection.
    -   Item authorization, min/max lot: from item tables (`authorized`, `minimum_lot`, `maximum_order`), by item_ref, after selection.
    -   Category min/max quantity: from category tables (`minimum_quantity`, `maximum_quantity`), by category, after selection.
    -   Item-category mapping: from item tables, by item_ref, after selection.
    -   Item usage per resource: from usage tables (`amount`, `unit`), by item_ref and resource, after selection.
    -   Resource capacity: sum of `amount` from capacity_ledger tables, by resource, after selection (convert units as needed).
    -   Incompatible pairs: from incompatible tables, by (item_a, item_b), after selection.
    -   Requires pairs: from requires tables, by (item_ref, prerequisite_ref), after selection.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (per-unit benefit * x[i]) 
    -   Minus sum over items: (item_fee for each item with x[i] > 0)
    -   Minus sum over categories: (category activation fee for each category with any x[i] > 0)
    -   Plus sum over bundles: (bundle bonus for each bundle where both items are ordered)
7.  **Formulate Constraints:**
    -   **Authorization and Lot Constraints:** For each item `i`:
        - If `authorized` == 0, enforce x[i] = 0.
        - If `authorized` > 0, enforce: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i], and x[i] is integer.
        - Link y[i] to x[i]: y[i] = 1 if x[i] > 0, else 0.
    -   **Category Quantity Constraints:** For each category `g`:
        - Sum of x[i] over all items in category g: minimum_quantity[g] ≤ sum_i_in_g x[i] ≤ maximum_quantity[g].
        - Link z[g] to x[i]: z[g] = 1 if any x[i] > 0 for i in g, else 0.
    -   **Resource Capacity Constraints:** For each resource `r`:
        - Sum over items: (usage per unit of r for i, converted to capacity_ledger units) * x[i] ≤ total available capacity for r (sum of capacity_ledger entries for r, after selection and unit conversion).
    -   **Incompatibility Constraints:** For each incompatible pair (i, j):
        - At most one of x[i], x[j] is positive: y[i] + y[j] ≤ 1.
    -   **Requires (Dependency) Constraints:** For each requires pair (i, k):
        - If x[i] > 0, then x[k] > 0 (prerequisite must be ordered in positive quantity): y[i] ≤ y[k].
    -   **Bundle Bonus Constraints:** For each bundle (item_a, item_b):
        - w[b] = 1 if both y[item_a] = 1 and y[item_b] = 1; else w[b] = 0.
    -   **Fixed Fee Application:** 
        - Item fee is charged once per item with x[i] > 0.
        - Category activation fee is charged once per category with any x[i] > 0.
        - Bundle bonus is awarded once per bundle if both items are ordered, zero otherwise (including if either is unauthorized).
    -   **Variable Domains:** 
        - x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] ∩ ℤ for authorized items; x[i] = 0 for unauthorized.
        - y[i], z[g], w[b] ∈ {0,1}.
    -   **Unit Conversion:** 
        - For resource usage and capacity, convert all units to match the capacity_ledger units (e.g., 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh) before summing or comparing.
[Abstract Model Plan END]