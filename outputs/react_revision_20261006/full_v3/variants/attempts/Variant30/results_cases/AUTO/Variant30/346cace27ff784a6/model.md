[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option incompatibilities and dependencies, bundle bonuses, and unit conversions for resource usage and capacity. Only the latest (highest revision) non-future, non-DELETE record for each (tenant, table, record_id) is used, and identical retransmissions are counted once.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (authorized options for NORTH as of 2026-03-12)
    - Categories `g` (categories associated with items)
    - Resources `r` (e.g., space, labor, power)
    - Bundles `b` (pairs of items eligible for a bundle bonus)
    - Incompatible pairs `(i, j)`
    - Requires pairs `(i, j)` (i requires j)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item/option `i`. Type: GRB.INTEGER.
    -   `y[i]` = 1 if item/option `i` is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered (category used), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are ordered (bundle bonus awarded), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of 'amount_cents' from 'benefit' tables, grouped by item_ref, after selection rules.
    -   Item activation fee: 'activation_fee_cents' from 'item_fee' tables, by item_ref.
    -   Category activation fee: 'activation_fee_cents' from 'category' tables, by category.
    -   Bundle bonus: 'bonus_cents' from 'bundle' tables, by (item_a, item_b) pair.
    -   Resource usage per unit: 'amount' and 'unit' from 'usage' tables, by item_ref and resource, with unit conversion (liter→ml, hour→minute, kwh→wh).
    -   Resource capacity: sum of 'amount' from 'capacity_ledger' tables, by resource, with unit conversion, after selection rules.
    -   Item authorization, min/max lot: 'authorized', 'minimum_lot', 'maximum_order' from 'item' tables, by item_ref.
    -   Category min/max quantity: 'minimum_quantity', 'maximum_quantity' from 'category' tables, by category.
    -   Incompatible pairs: from 'incompatible' tables, as (item_a, item_b) pairs.
    -   Requires pairs: from 'requires' tables, as (item_ref, prerequisite_ref) pairs.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (per-unit benefit) × x[i]
    -   Minus: sum of item activation fees for each item with x[i] > 0 (charged once per used item)
    -   Minus: sum of category activation fees for each category with any item ordered (charged once per used category)
    -   Plus: sum of bundle bonuses for each bundle where both items are ordered (awarded once per bundle)
7.  **Formulate Constraints:**
    -   **Data Selection:** For each table, select only rows for tenant = NORTH, effective_date ≤ 2026-03-12, and for each (table, record_id), keep only the row with the highest revision (excluding DELETE actions). Identical retransmissions count once.
    -   **Authorization:** For each item, x[i] = 0 if 'authorized' = 0; otherwise, x[i] ≥ 0.
    -   **Lot Size:** For each item, if x[i] > 0, then minimum_lot[i] ≤ x[i] ≤ maximum_order[i]; x[i] = 0 or x[i] ≥ minimum_lot[i].
    -   **Resource Limits:** For each resource r, sum over items of (resource usage per unit, converted to base units) × x[i] ≤ total available capacity for r (converted to base units).
    -   **Category Quantity Limits:** For each category g, sum of x[i] over items in g must satisfy minimum_quantity[g] ≤ total ≤ maximum_quantity[g] if any item in g is ordered; if no item in g is ordered, total = 0.
    -   **Category Activation Fee Linking:** z[g] = 1 if any x[i] > 0 for i in g; z[g] = 0 otherwise.
    -   **Item Activation Fee Linking:** y[i] = 1 if x[i] > 0; y[i] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i, j), at most one of x[i], x[j] can be positive: y[i] + y[j] ≤ 1.
    -   **Requires Dependency:** For each requires pair (i, j), x[i] > 0 ⇒ x[j] > 0 (i.e., y[i] ≤ y[j]).
    -   **Bundle Bonus Linking:** For each bundle (i, j), w[b] = 1 if both x[i] > 0 and x[j] > 0; w[b] = 0 otherwise.
    -   **Integrality:** All x[i] are integer, y[i], z[g], w[b] are binary.
    -   **Unit Conversion:** Ensure all resource usage and capacity are compared in the same base units (e.g., 1 liter = 1000 ml, 1 hour = 60 minutes, 1 kwh = 1000 wh).
[Abstract Model Plan END]