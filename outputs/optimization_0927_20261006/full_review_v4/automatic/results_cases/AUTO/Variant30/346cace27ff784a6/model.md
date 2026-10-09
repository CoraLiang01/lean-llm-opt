[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The model must account for per-unit benefits, item and category activation fees, bundle bonuses, resource and category quantity limits, minimum/maximum order sizes, authorization, incompatibilities, and requires dependencies. All data tables must be filtered per the specified as-of/revision/deletion rule before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, after filtering for NORTH and as-of rules)
    - Categories `g` (from category tables, after filtering)
    - Resources `r` (from usage/capacity tables)
    - Bundles `b` (from bundle tables)
    - Incompatible pairs `(i,j)` (from incompatible tables)
    - Requires pairs `(i,k)` (from requires tables)
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER.
    - `z[i]` = Binary, 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    - `y[g]` = Binary, 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    - `w[b]` = Binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Per-unit benefit: sum of `amount_cents` from all benefit table rows for each item `i` (after filtering).
    - Item activation fee: `activation_fee_cents` from item_fee table for each item `i` (after filtering).
    - Category activation fee: `activation_fee_cents` from category table for each category `g` (after filtering).
    - Bundle bonus: `bonus_cents` from bundle table for each bundle `b` (after filtering).
    - Resource usage per unit: `amount` and `unit` from usage table for each item-resource pair (after filtering).
    - Resource capacity: sum of `amount` (converted to common units) from capacity_ledger table for each resource `r` (after filtering).
    - Item authorization, minimum_lot, maximum_order, category: from item table for each item `i` (after filtering).
    - Category minimum_quantity, maximum_quantity: from category table for each category `g` (after filtering).
    - Incompatible pairs: from incompatible table (after filtering).
    - Requires pairs: from requires table (after filtering).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (per-unit benefit) × `q[i]`
    - Minus sum over items: (item activation fee) × `z[i]` (only if `q[i]>0`)
    - Minus sum over categories: (category activation fee) × `y[g]` (only if any item in `g` is selected)
    - Plus sum over bundles: (bundle bonus) × `w[b]` (only if both items in bundle are selected)
7.  **Formulate Constraints:**
    - **Item selection and order size:**
        - For each item `i`: `q[i] = 0` if not authorized; else `minimum_lot[i] ≤ q[i] ≤ maximum_order[i]` or `q[i]=0`.
        - For each item `i`: `z[i] = 1` iff `q[i] > 0`; enforce with `q[i] ≥ minimum_lot[i] * z[i]` and `q[i] ≤ maximum_order[i] * z[i]`.
    - **Category quantity limits:**
        - For each category `g`: `minimum_quantity[g] ≤ sum_{i in g} q[i] ≤ maximum_quantity[g]`.
        - For each item `i` in category `g`: `z[i] ≤ y[g]`; for each category `g`: `y[g] ≤ sum_{i in g} z[i]`.
    - **Resource capacity:**
        - For each resource `r`: sum over items of (resource usage per unit, converted to common units) × `q[i]` ≤ (resource capacity, in same units).
    - **Incompatibility:**
        - For each incompatible pair `(i,j)`: `z[i] + z[j] ≤ 1`.
    - **Requires dependencies:**
        - For each requires pair `(i,k)`: `z[i] ≤ z[k]` and `q[k] ≥ minimum_lot[k] * z[i]` (i.e., if `i` is selected, `k` must be selected with positive quantity).
    - **Bundle bonuses:**
        - For each bundle `b` with items `(i,j)`: `w[b] ≤ z[i]`, `w[b] ≤ z[j]`, `w[b] ≥ z[i] + z[j] - 1` (i.e., `w[b]=1` iff both items are selected).
    - **Variable domains:**
        - All `q[i]` are integer and ≥ 0; all `z[i]`, `y[g]`, `w[b]` are binary.
[Abstract Model Plan END]