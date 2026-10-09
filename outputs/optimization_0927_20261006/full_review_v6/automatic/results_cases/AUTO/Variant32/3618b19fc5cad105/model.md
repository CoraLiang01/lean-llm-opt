[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, as of 2026-05-07, the vehicle order for the Oslo dealership that yields the largest net benefit in USD cents, considering multi-currency benefit components, item and category fees, bundle bonuses, resource and category limits, option authorization, incompatibilities, and requires dependencies. All data must be filtered per a strict as-of/revision/deletion rule for each table before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, logical dependencies, and resource constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`I`): All valid OSLO_NEW_CARS items as of 2026-05-07.
    - Categories (`C`): All valid OSLO_NEW_CARS categories as of 2026-05-07.
    - Resources (`R`): All valid OSLO_NEW_CARS resources as of 2026-05-07.
    - Bundles (`B`): All valid OSLO_NEW_CARS bundle pairs as of 2026-05-07.
    - Incompatible pairs (`P`): All valid OSLO_NEW_CARS incompatible item pairs as of 2026-05-07.
    - Requires pairs (`Q`): All valid OSLO_NEW_CARS requires (item, prerequisite) pairs as of 2026-05-07.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item/option `i` (must be zero if unauthorized; otherwise, between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = Binary flag: 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `y[c]` = Binary flag: 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary flag: 1 if both items in bundle pair `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit components: From `benefit` table, sum of all signed amounts per item, converted to USD cents using the latest as-of-2026-05-07 FX rates (`fx` table: usd_cents_numerator/denominator).
    -   Item fees: From `item_fee` table, activation_fee_cents per item.
    -   Category limits and activation fees: From `category` table, minimum_quantity, maximum_quantity, activation_fee_cents per category.
    -   Item authorization, lot/order bounds, and category mapping: From `item` table, authorized, minimum_lot, maximum_order, category.
    -   Resource usage per unit: From `usage` table, amount and unit per item-resource pair (convert all to ledger units: 1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    -   Resource capacity: From `capacity_ledger` table, sum of all signed amounts per resource in ledger units.
    -   Bundle bonuses: From `bundle` table, bonus_cents per bundle pair.
    -   Incompatible pairs: From `incompatible` table, all item pairs.
    -   Requires dependencies: From `requires` table, all (item, prerequisite) pairs.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (per-unit USD cents benefit) × (ordered quantity)
    -   Minus: sum of item_fee for each item with positive quantity (once per item)
    -   Minus: sum of category activation_fee for each category with any item selected (once per category)
    -   Plus: sum of bundle bonus for each bundle where both items are selected (once per bundle)
7.  **Formulate Constraints:**
    -   **Item authorization and bounds:** For each item, if authorized, quantity is integer in [minimum_lot, maximum_order] or zero; if unauthorized, quantity must be zero. Enforce: `minimum_lot * z[i] <= q[i] <= maximum_order * z[i]`; if unauthorized, `q[i] = 0` and `z[i] = 0`.
    -   **Category quantity limits:** For each category, sum of quantities of its items must be within [minimum_quantity, maximum_quantity], unconditionally (even if no items are selected).
    -   **Category activation flag:** For each category, `y[c]` = 1 iff any item in category `c` is selected: for all items in `c`, `z[i] <= y[c]`, and `y[c] <= sum(z[i] for i in c)`.
    -   **Resource capacity:** For each resource, total usage (sum over items of per-unit usage × quantity, all in ledger units) ≤ signed total capacity from capacity_ledger (in same units).
    -   **Incompatible pairs:** For each incompatible pair (i, j), at most one can be selected: `z[i] + z[j] <= 1`.
    -   **Requires dependencies:** For each (item, prerequisite) pair, if item is selected (`z[i]=1`), then prerequisite must have positive quantity: `q[prerequisite] >= z[i]`.
    -   **Bundle bonus flags:** For each bundle (i, j), `w[b] = 1` iff both `z[i] = 1` and `z[j] = 1`: `w[b] <= z[i]`, `w[b] <= z[j]`, `w[b] >= z[i] + z[j] - 1`.
    -   **Item fee flags:** For each item, item_fee is charged once if `z[i]=1`.
    -   **Category activation fee flags:** For each category, activation_fee is charged once if `y[c]=1`.
    -   **Bundle bonus flags:** For each bundle, bonus is awarded once if `w[b]=1`, zero otherwise (including if either item is unauthorized).
    -   **Integrality:** All quantities and flags are integer/binary as defined.
[Abstract Model Plan END]