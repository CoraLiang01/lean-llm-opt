[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, as of 2026-05-07, the vehicle order for the Oslo dealership that yields the largest net benefit in USD cents, subject to a complex set of eligibility, resource, category, compatibility, dependency, and fee/bonus rules. All data must be filtered per a strict as-of/revision/deletion rule for each logical table before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, compatibility, and dependency logic.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`I`): All valid OSLO_NEW_CARS items after as-of filtering.
    - Categories (`C`): All valid OSLO_NEW_CARS categories after as-of filtering.
    - Resources (`R`): All valid OSLO_NEW_CARS resources after as-of filtering.
    - Bundles (`B`): All valid OSLO_NEW_CARS bundles after as-of filtering.
    - Incompatible pairs (`P`): All valid OSLO_NEW_CARS incompatible item pairs after as-of filtering.
    - Requires pairs (`Q`): All valid OSLO_NEW_CARS requires (dependency) pairs after as-of filtering.
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity ordered of item/option `i` (must be zero if unauthorized). Type: GRB.INTEGER.
    - `z[i]` = Binary, 1 if item/option `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = Binary, 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    - `w[b]` = Binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Benefit per unit (USD cents): For each item, sum all valid benefit components (from `benefit` table), converting each `amount` to USD cents using the corresponding `fx` table entry (`usd_cents_numerator`/`denominator`), after as-of filtering.
    - Item activation fee (USD cents): From `item_fee` table, per item, after as-of filtering.
    - Category activation fee (USD cents): From `category` table, per category, after as-of filtering.
    - Bundle bonus (USD cents): From `bundle` table, per bundle, after as-of filtering.
    - Minimum/maximum lot/order: From `item` table, per item, after as-of filtering.
    - Authorization: From `item` table, per item, after as-of filtering.
    - Category membership: From `item` table, per item, after as-of filtering.
    - Category quantity limits: From `category` table, per category, after as-of filtering.
    - Resource usage per unit: From `usage` table, per item-resource pair, after as-of filtering; convert all usage and capacity to the same units (1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    - Resource capacity: From `capacity_ledger` table, sum all valid entries per resource, after as-of filtering and unit conversion.
    - Incompatible pairs: From `incompatible` table, after as-of filtering.
    - Requires pairs: From `requires` table, after as-of filtering.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    - Sum over all items: (benefit per unit in USD cents) × `q[i]`
    - Minus: sum of item activation fees for each item with `q[i] > 0` (charge once per item with positive quantity)
    - Minus: sum of category activation fees for each category with at least one selected item (charge once per used category)
    - Plus: sum of bundle bonuses for each bundle where both items are selected (award once per qualifying bundle)
7.  **Formulate Constraints:**
    - **Authorization:** For each item, `q[i] = 0` if not authorized; if authorized, `q[i]` ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
    - **Item selection linking:** For each item, `z[i] = 1` iff `q[i] > 0`; enforce `q[i] ≥ minimum_lot[i] * z[i]` and `q[i] ≤ maximum_order[i] * z[i]`, and `q[i] = 0` if `z[i] = 0`.
    - **Category selection linking:** For each category, `y[c] = 1` iff at least one item in category `c` is selected; enforce `z[i] ≤ y[c]` for all items `i` in category `c`.
    - **Category quantity limits:** For each category, sum of `q[i]` over all items in category `c` must be within [minimum_quantity[c], maximum_quantity[c]] (unconditional, applies even if no items are selected).
    - **Resource capacity:** For each resource, sum over all items of (resource usage per unit, in common units) × `q[i]` ≤ total available capacity (in same units).
    - **Incompatible pairs:** For each incompatible pair (i, j), `z[i] + z[j] ≤ 1` (cannot select both).
    - **Requires dependencies:** For each requires pair (i, j), `q[i] > 0` ⇒ `q[j] > 0` (i.e., `z[i] ≤ z[j]`), but no ratio constraint.
    - **Bundle selection linking:** For each bundle (i, j), `w[b] = 1` iff both `z[i] = 1` and `z[j] = 1`; enforce `w[b] ≤ z[i]`, `w[b] ≤ z[j]`, `w[b] ≥ z[i] + z[j] - 1`.
    - **Item and category activation fees:** Deduct item activation fee once per item with `q[i] > 0`; deduct category activation fee once per category with `y[c] = 1`.
    - **Bundle bonuses:** Award bundle bonus once per bundle with `w[b] = 1`, zero otherwise (including if either item is unauthorized).
    - **Integrality:** All `q[i]` are integer, all `z[i]`, `y[c]`, `w[b]` are binary.
[Abstract Model Plan END]