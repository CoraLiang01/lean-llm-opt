[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, using only records effective on or before 2026-03-12, in order to maximize net benefit (in USD cents). The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, minimum/maximum order sizes, option incompatibilities and dependencies, bundle bonuses, and unit conversions for resource usage and capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from the filtered item tables for NORTH)
    - Categories `g` (from the filtered category tables for NORTH)
    - Resources `r` (from the filtered resource/capacity tables for NORTH)
    - Bundles (pairs of items eligible for a bonus)
    - Incompatible pairs (item pairs that cannot both be ordered)
    - Requires pairs (item pairs where one requires the other)
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER.
    -   `z[i]` = 1 if item `i` is ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `w[g]` = 1 if any item in category `g` is ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: sum of `amount_cents` from all benefit rows for each item (from benefit tables, filtered for NORTH and effective date).
    -   Item activation fee: `activation_fee_cents` from item_fee tables (filtered for NORTH and effective date).
    -   Category activation fee: `activation_fee_cents` from category tables (filtered for NORTH and effective date).
    -   Minimum/maximum order: `minimum_lot`, `maximum_order` from item tables (filtered for NORTH and effective date).
    -   Authorization: `authorized` from item tables (filtered for NORTH and effective date).
    -   Category quantity limits: `minimum_quantity`, `maximum_quantity` from category tables (filtered for NORTH and effective date).
    -   Resource usage per unit: `amount` and `unit` from usage tables (filtered for NORTH and effective date).
    -   Resource capacity: sum of `amount` (with sign) from capacity_ledger tables (filtered for NORTH and effective date), with unit conversion as needed.
    -   Incompatible pairs: from incompatible tables (filtered for NORTH and effective date).
    -   Requires pairs: from requires tables (filtered for NORTH and effective date).
    -   Bundle bonuses: `bonus_cents` from bundle tables (filtered for NORTH and effective date).
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over items: (per-unit benefit) × (quantity ordered)
    -   Minus: sum of item activation fees for each item with positive quantity
    -   Minus: sum of category activation fees for each category with any item ordered
    -   Plus: sum of bundle bonuses for each bundle where both items are ordered
7.  **Formulate Constraints:**
    -   **Item Authorization:** For each item, if `authorized` is zero, force `q[i] = 0`.
    -   **Item Order Bounds:** For each item, `minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]`, and `q[i] ≥ 0`, integer.
    -   **Item Activation Indicator:** For each item, `z[i] = 1` if `q[i] > 0`, else `z[i] = 0`.
    -   **Category Quantity Bounds:** For each category, sum of `q[i]` over items in category `g` must satisfy `minimum_quantity[g] * w[g] ≤ sum_{i in g} q[i] ≤ maximum_quantity[g] * w[g]`.
    -   **Category Activation Indicator:** For each category, `w[g] = 1` if any `q[i] > 0` for `i` in `g`, else `w[g] = 0`.
    -   **Resource Capacity:** For each resource, sum over items of (resource usage per unit, converted to capacity units) × `q[i]` ≤ (signed total capacity for that resource, converted to same units).
    -   **Incompatibility:** For each incompatible pair `(i, j)`, enforce `z[i] + z[j] ≤ 1`.
    -   **Requires Dependency:** For each requires pair `(i, j)`, enforce `q[i] > 0 ⇒ q[j] > 0` (can be modeled as `q[i] ≤ M * z[j]` for large M, or `z[i] ≤ z[j]`).
    -   **Bundle Bonus:** For each bundle `(i, j)`, set `b[bundle] = 1` if both `q[i] > 0` and `q[j] > 0`, else `b[bundle] = 0`. Award bonus only if both are ordered and authorized.
    -   **Identical Retransmissions:** For all tables, before joining or summing, select only the highest integer revision for each (tenant, table, record_id) with effective_date ≤ 2026-03-12, and discard if that revision is DELETE. Identical retransmissions count once.
    -   **Unconditional Category/Resource Bounds:** Apply category and resource quantity/capacity bounds unconditionally, regardless of activation.
    -   **Unit Conversion:** When comparing resource usage and capacity, convert all units to a common base (e.g., 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh).
[Abstract Model Plan END]