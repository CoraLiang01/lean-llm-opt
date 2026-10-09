[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, that maximize net benefit (in USD cents). The model must account for per-unit benefit, item and category activation fees, bundle bonuses, resource and category quantity limits, minimum/maximum order sizes, authorization, incompatibilities, and requires dependencies. All data tables must be filtered for NORTH, as-of 2026-03-12, using the highest integer revision per (tenant, table, record_id), discarding records with DELETE as the latest action, and counting identical retransmissions only once.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, after filtering)
    - Categories `g` (from category tables, after filtering)
    - Resources `r` (from usage/capacity tables, after filtering)
    - Bundles `b` (from bundle tables, after filtering)
    - Incompatible pairs `(i,j)` (from incompatible tables, after filtering)
    - Requires pairs `(i,k)` (from requires tables, after filtering)
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER.
    -   `z[i]` = 1 if item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for item `i`: sum of all `amount_cents` from benefit tables for item_ref = i (after filtering).
    -   Item activation fee for item `i`: `activation_fee_cents` from item_fee tables (after filtering).
    -   Category activation fee for category `g`: `activation_fee_cents` from category tables (after filtering).
    -   Bundle bonus for bundle `b`: `bonus_cents` from bundle tables (after filtering).
    -   Minimum/maximum order for item `i`: `minimum_lot`, `maximum_order` from item tables (after filtering).
    -   Authorization for item `i`: `authorized` from item tables (after filtering); unauthorized items must have q[i]=0.
    -   Category quantity limits: `minimum_quantity`, `maximum_quantity` from category tables (after filtering).
    -   Resource usage per unit for item `i` and resource `r`: `amount` and `unit` from usage tables (after filtering).
    -   Resource capacity for resource `r`: sum of all `amount` (with sign) from capacity_ledger tables for resource `r` (after filtering), converted to the same unit as usage (using 1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    -   Incompatible pairs: from incompatible tables (after filtering).
    -   Requires pairs: from requires tables (after filtering).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (per-unit benefit[i] * q[i])
    -   Minus: sum over selected items: item activation_fee_cents[i] * z[i] (only if q[i]>0)
    -   Minus: sum over used categories: category activation_fee_cents[g] * y[g] (only if any item in g is selected)
    -   Plus: sum over bundles: bundle_bonus_cents[b] * w[b] (only if both items in bundle b are selected)
7.  **Formulate Constraints:**
    -   **Item selection and order size:**
        - For each item i:
            - If authorized[i] > 0: minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; q[i] ≥ 0 integer; z[i] ∈ {0,1}
            - If authorized[i] = 0: q[i] = 0; z[i] = 0
    -   **Category quantity limits:**
        - For each category g: minimum_quantity[g] ≤ sum_{i in g} q[i] ≤ maximum_quantity[g]
        - For each item i in category g: z[i] ≤ y[g]; y[g] ≤ sum_{i in g} z[i]
    -   **Resource capacity:**
        - For each resource r: sum_{i} (usage_per_unit[i,r] * q[i]) ≤ total_capacity[r] (all in common units)
    -   **Incompatibility:**
        - For each incompatible pair (i,j): z[i] + z[j] ≤ 1
    -   **Requires dependencies:**
        - For each requires pair (i,k): q[i] ≤ maximum_order[i] * z[i]; q[k] ≥ z[i]
    -   **Bundle bonuses:**
        - For each bundle b with items (i,j): w[b] ≤ z[i]; w[b] ≤ z[j]; w[b] ≥ z[i] + z[j] - 1
    -   **Variable domains:** All variables as above; all quantities integer and nonnegative; all flags binary.
[Abstract Model Plan END]