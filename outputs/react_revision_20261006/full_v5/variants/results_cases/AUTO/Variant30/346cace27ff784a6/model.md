[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option incompatibilities and dependencies, bundle bonuses, and unit conversions for resource usage and capacity. Only the latest (highest revision) non-future, non-DELETE record for each (tenant, table, record_id) is used, and identical retransmissions are counted once.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All item_ref for authorized options in NORTH, as of 2026-03-12.
    - Categories (`g`): All categories present in the selected items.
    - Resources (`r`): All resources referenced in usage/capacity tables (e.g., labor, space, power).
    - Bundles (`b`): All bundle pairs (item_a, item_b) for which both items are present and authorized.
    - Incompatibility pairs (`(i,j)`): All pairs of items that are incompatible.
    - Requires pairs (`(i,j)`): All pairs where item i requires item j.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered for item i (must be 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is ordered (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is ordered (category is used), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are ordered (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: Sum of amount_cents from 'benefit' tables for each item_ref (after selection rules).
    -   Item activation fee: activation_fee_cents from 'item_fee' tables for each item_ref.
    -   Category activation fee: activation_fee_cents from 'category' tables for each category.
    -   Minimum/maximum lot: minimum_lot and maximum_order from 'item' tables for each item_ref.
    -   Authorization: authorized from 'item' tables for each item_ref (must be >0 to allow ordering).
    -   Category quantity limits: minimum_quantity and maximum_quantity from 'category' tables for each category.
    -   Resource usage per unit: amount and unit from 'usage' tables for each item_ref and resource.
    -   Resource capacity: sum of amount (with sign) and unit from 'capacity_ledger' tables for each resource (convert all to base units).
    -   Incompatibility: item_a, item_b pairs from 'incompatible' tables.
    -   Requires: item_ref, prerequisite_ref pairs from 'requires' tables.
    -   Bundle bonus: bonus_cents from 'bundle' tables for each (item_a, item_b) pair.
    -   Bundle eligibility: both items must be authorized and ordered.
    -   Unit conversions: 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (per-unit benefit) * x[i]
    -   Minus: sum of item activation fees for each item with x[i] > 0 (charge once per item)
    -   Minus: sum of category activation fees for each category with any x[i] > 0 in that category (charge once per category)
    -   Plus: sum of bundle bonuses for each bundle where both items are ordered (award once per bundle)
7.  **Formulate Constraints:**
    -   **Data selection constraint:** For each table, only use the latest (highest revision) non-future (effective_date ≤ 2026-03-12), non-DELETE record for each (tenant, table, record_id). Discard records with action DELETE at the highest revision. Identical retransmissions count once.
    -   **Authorization constraint:** For each item i, if authorized = 0, then x[i] = 0.
    -   **Lot size constraints:** For each item i, if x[i] > 0, then minimum_lot[i] ≤ x[i] ≤ maximum_order[i]; else x[i] = 0.
    -   **Category quantity constraints:** For each category g, sum of x[i] over all items in g must satisfy minimum_quantity[g] ≤ sum_i_in_g x[i] ≤ maximum_quantity[g].
    -   **Category activation constraint:** z[g] = 1 if any x[i] > 0 for i in g; z[g] = 0 otherwise.
    -   **Resource capacity constraints:** For each resource r, sum over items of (resource usage per unit, converted to base units) * x[i] ≤ total available capacity for r (sum of capacity_ledger entries, converted to base units).
    -   **Item activation constraint:** y[i] = 1 if x[i] > 0; y[i] = 0 otherwise.
    -   **Incompatibility constraints:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1 (cannot order both).
    -   **Requires constraints:** For each requires pair (i, j), y[i] ≤ y[j] and x[j] ≥ 1 if x[i] ≥ 1 (i.e., cannot order i unless j is also ordered in positive quantity).
    -   **Bundle bonus eligibility:** For each bundle (i, j), w[b] = 1 if y[i] = 1 and y[j] = 1; w[b] = 0 otherwise.
    -   **Integrality constraints:** All x[i] are integer, y[i], z[g], w[b] are binary.
    -   **Non-negativity:** All x[i] ≥ 0.
[Abstract Model Plan END]