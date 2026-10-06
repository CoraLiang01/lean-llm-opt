[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option incompatibilities and dependencies, bundle bonuses, and unit conversions for resource usage and capacity. Only the latest (highest revision) non-future, non-DELETE record for each (tenant, table, record_id) is used, and identical retransmissions are counted once.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (i): All item_ref values for authorized options in NORTH, as of 2026-03-12, after applying the selection rule.
    - Categories (g): All categories associated with selected items.
    - Resources (r): All resources (e.g., labor, space, power) referenced in usage/capacity tables.
    - Bundles (b): All bundle pairs (item_a, item_b) for which both items are in the selected set.
    - Incompatibility pairs (i,j): All pairs of items that are incompatible.
    - Requires pairs (i,j): All pairs where item i requires item j.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered for item i (must be 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = Binary variable: 1 if item i is ordered (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = Binary variable: 1 if any item in category g is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary variable: 1 if both items in bundle b are ordered, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of amount_cents from 'benefit' tables, grouped by item_ref, after selection rule.
    -   Item activation fee: activation_fee_cents from 'item_fee' tables, by item_ref, after selection rule.
    -   Category activation fee: activation_fee_cents from 'category' tables, by category, after selection rule.
    -   Bundle bonus: bonus_cents from 'bundle' tables, by (item_a, item_b), after selection rule.
    -   Resource usage per unit: amount and unit from 'usage' tables, by item_ref and resource, after selection rule.
    -   Resource capacity: sum of amount (with sign) and unit from 'capacity_ledger' tables, by resource, after selection rule.
    -   Item authorization, min/max lot: authorized, minimum_lot, maximum_order from 'item' tables, by item_ref, after selection rule.
    -   Category min/max quantity: minimum_quantity, maximum_quantity from 'category' tables, by category, after selection rule.
    -   Incompatibility pairs: (item_a, item_b) from 'incompatible' tables, after selection rule.
    -   Requires pairs: (item_ref, prerequisite_ref) from 'requires' tables, after selection rule.
    -   Unit conversions: 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (per-unit benefit[i] * x[i])
    -   Minus sum over items: (item activation_fee[i] * y[i]) for each item with x[i] > 0
    -   Minus sum over categories: (category activation_fee[g] * z[g]) for each category used
    -   Plus sum over bundles: (bundle_bonus[b] * w[b]) for each bundle where both items are ordered
7.  **Formulate Constraints:**
    -   **Item authorization and lot constraints:** For each item i:
        - If authorized[i] == 0, x[i] = 0.
        - If authorized[i] > 0, minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; y[i] ∈ {0,1}.
    -   **Category quantity limits:** For each category g:
        - minimum_quantity[g] * z[g] ≤ sum_{i in g} x[i] ≤ maximum_quantity[g] * z[g]; z[g] ∈ {0,1}.
    -   **Resource capacity constraints:** For each resource r:
        - sum_{i} (resource_usage_per_unit[i,r] * x[i] * unit_conversion) ≤ resource_capacity[r] (after unit conversion and summing all relevant ledger entries).
    -   **Category activation linking:** For each category g:
        - z[g] ≥ y[i] for all items i in category g (if any item in g is ordered, category is activated).
    -   **Item activation linking:** For each item i:
        - y[i] ≥ 1 if x[i] > 0; y[i] = 0 if x[i] = 0.
    -   **Incompatibility constraints:** For each incompatible pair (i,j):
        - y[i] + y[j] ≤ 1 (cannot order both incompatible items).
    -   **Requires constraints:** For each requires pair (i,j):
        - x[i] ≤ maximum_order[i] * y[j] (if x[i] > 0, then x[j] > 0).
    -   **Bundle bonus linking:** For each bundle b = (i,j):
        - w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1 (w[b] = 1 iff both items are ordered).
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i], z[g], w[b] ∈ {0,1}.
    -   **All constraints and parameters are based only on the selected (latest, non-future, non-DELETE) records for tenant NORTH as of 2026-03-12, per the selection rule.**
[Abstract Model Plan END]