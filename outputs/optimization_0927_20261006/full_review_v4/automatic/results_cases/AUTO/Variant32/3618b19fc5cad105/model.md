[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, as of 2026-05-07, the vehicle order for the Oslo dealership that yields the largest net benefit in USD cents, considering all item, category, resource, compatibility, dependency, fee, and bonus rules. All data must be filtered per the specified as-of/revision/deletion/retransmission logic for each table before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, compatibility, and dependency logic.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle options) available for order (`i ∈ Items`)
    - Categories (`g ∈ Categories`)
    - Resources (`r ∈ Resources`)
    - Bundles (bonus pairs) (`b ∈ Bundles`)
    - Incompatible pairs (`(i,j) ∈ Incompatibles`)
    - Requires pairs (`(i,k) ∈ Requires`)
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER, with bounds per item.
    - `z[i]` = Binary, 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    - `y[g]` = Binary, 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    - `w[b]` = Binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Item benefit components: from `benefit` table, sum of all signed `amount` fields per item, converted to USD cents using `fx` table (`amount * usd_cents_numerator / denominator`).
    - Item fees: from `item_fee` table, `activation_fee_cents` per item.
    - Category limits and activation fees: from `category` table, `minimum_quantity`, `maximum_quantity`, `activation_fee_cents` per category.
    - Item bounds and authorization: from `item` table, `minimum_lot`, `maximum_order`, `authorized` (must be >0 for eligibility).
    - Resource usage per unit: from `usage` table, `amount` per item-resource, converted to canonical units (ml, minute, wh).
    - Resource capacities: from `capacity_ledger` table, sum of all signed `amount` per resource, in canonical units.
    - Incompatibilities: from `incompatible` table, pairs of items that cannot both be selected.
    - Requires dependencies: from `requires` table, pairs where selection of one item requires positive quantity of another.
    - Bundle bonuses: from `bundle` table, `bonus_cents` per eligible pair.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - For each item: sum of all benefit components per unit (in USD cents) × quantity ordered.
    - Subtract item fee once for each item with positive quantity.
    - Subtract category activation fee once for each category used.
    - Add bundle bonus once for each bundle where both items are selected.
7.  **Formulate Constraints:**
    - **Item selection and bounds:** For each item `i`, `q[i] = 0` if not authorized; else, `minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]`, and `q[i] = 0` if not authorized.
    - **Category quantity limits:** For each category `g`, sum of `q[i]` over items in `g` must satisfy `minimum_quantity[g] ≤ sum(q[i] for i in g) ≤ maximum_quantity[g]`.
    - **Category activation:** For each category `g`, `y[g] = 1` iff any `z[i] = 1` for `i` in `g`; else `y[g] = 0`.
    - **Resource capacity:** For each resource `r`, sum over items of (resource usage per unit in canonical units × `q[i]`) ≤ total available capacity for `r` (in canonical units).
    - **Incompatibility:** For each incompatible pair `(i,j)`, `z[i] + z[j] ≤ 1`.
    - **Requires dependencies:** For each requires pair `(i,k)`, `q[i] > 0` ⇒ `q[k] > 0` (enforced as `q[i] ≤ maximum_order[i] * (q[k] ≥ 1)` or equivalent).
    - **Bundle bonus eligibility:** For each bundle `b` with items `(i,j)`, `w[b] ≤ z[i]`, `w[b] ≤ z[j]`, `w[b] ≥ z[i] + z[j] - 1`.
    - **Item fee application:** For each item, item fee is charged once if `q[i] > 0`.
    - **Category activation fee application:** For each category, activation fee is charged once if `y[g] = 1`.
    - **Bundle bonus application:** For each bundle, bonus is awarded once if `w[b] = 1`, zero otherwise.
    - **Non-negativity and integrality:** All `q[i]` ≥ 0, integer; all `z[i]`, `y[g]`, `w[b]` binary.
[Abstract Model Plan END]