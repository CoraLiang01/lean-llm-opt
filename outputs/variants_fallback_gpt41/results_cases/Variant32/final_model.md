[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, as of 2026-05-07, the vehicle order for the Oslo dealership that yields the largest net benefit in USD cents, subject to a complex set of eligibility, resource, category, compatibility, dependency, and fee/bonus rules. All data must be filtered by a precise as-of/revision/deletion rule for each table before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, compatibility, and dependency logic.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`I`): All valid OSLO_NEW_CARS items after as-of filtering.
    - Categories (`C`): All valid OSLO_NEW_CARS categories after as-of filtering.
    - Resources (`R`): All valid OSLO_NEW_CARS resources after as-of filtering.
    - Bundles (`B`): All valid OSLO_NEW_CARS bundles after as-of filtering.
    - Incompatible pairs (`P`): All valid OSLO_NEW_CARS incompatible item pairs after as-of filtering.
    - Requires pairs (`Q`): All valid OSLO_NEW_CARS requires (dependency) pairs after as-of filtering.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item/option `i` (within [minimum_lot, maximum_order] if authorized, 0 if unauthorized). Type: GRB.INTEGER.
    -   `z[i]` = Binary, 1 if item/option `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[c]` = Binary, 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit (USD cents): For each item, sum all benefit components (from `benefit` table) after converting each amount to USD cents using the latest valid FX rate (from `fx` table), as-of 2026-05-07, for the item's currency.
    -   Item fee (USD cents): From `item_fee` table, per item, as-of 2026-05-07.
    -   Category activation fee (USD cents): From `category` table, per category, as-of 2026-05-07.
    -   Bundle bonus (USD cents): From `bundle` table, per bundle, as-of 2026-05-07.
    -   Minimum lot, maximum order, authorized: From `item` table, per item, as-of 2026-05-07.
    -   Category membership: From `item` table, per item, as-of 2026-05-07.
    -   Resource usage per unit: From `usage` table, per item-resource, as-of 2026-05-07, converted to canonical units (liter→ml, hour→minute, kwh→wh).
    -   Resource capacity: From `capacity_ledger` table, sum all signed entries per resource, as-of 2026-05-07, in canonical units.
    -   Category quantity limits: From `category` table, per category, as-of 2026-05-07.
    -   Incompatible pairs: From `incompatible` table, as-of 2026-05-07.
    -   Requires dependencies: From `requires` table, as-of 2026-05-07.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (benefit per unit in USD cents) × q[i]
    -   Minus: sum of item_fee[i] for each item with q[i] > 0 (deducted once per selected item)
    -   Minus: sum of category activation_fee[c] for each category with at least one selected item (deducted once per used category)
    -   Plus: sum of bundle bonus for each bundle where both items are selected (awarded once per such bundle)
7.  **Formulate Constraints:**
    -   **Item eligibility and bounds:** For each item, if authorized > 0, q[i] ∈ {0} ∪ [minimum_lot, maximum_order]; if not authorized, q[i] = 0. Enforce z[i] = 1 iff q[i] > 0.
    -   **Category quantity limits:** For each category, sum of q[i] over items in category c is within [minimum_quantity, maximum_quantity] (unconditional).
    -   **Category activation:** For each category, y[c] = 1 iff any z[i] = 1 for i in c; y[c] = 0 otherwise.
    -   **Resource capacity:** For each resource, sum over items of (resource usage per unit in canonical units) × q[i] ≤ total available capacity (in canonical units).
    -   **Incompatible pairs:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   **Requires dependencies:** For each requires pair (i, j), q[i] > 0 ⇒ q[j] > 0 (i.e., z[i] ≤ z[j]).
    -   **Bundle activation:** For each bundle (i, j), w[b] = 1 iff z[i] = 1 and z[j] = 1; w[b] = 0 otherwise.
[Abstract Model Plan END]