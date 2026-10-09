[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource capacity, category quantity bounds, item authorization, lot size/order limits, incompatibility, and dependency (requires) constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, activation, and logical (pairwise and dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the item table.
    - Categories from the category table.
    - Resources from the resource/capacity_ledger/usage tables.
    - Bundles (item pairs with bonuses) from the bundle table.
    - Incompatible pairs from the incompatible table.
    - Requires pairs from the requires table.
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity of item `i` selected for delivery (0 if not selected). Type: GRB.INTEGER.
    - `z[i]` = 1 if item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    - `b[p]` = 1 if both items in bundle pair `p` are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - `unit_benefit_cents[i]` (from item table): per-unit benefit for item `i`.
        - `item_fee_cents[i]` (from item table): fixed fee if any of item `i` is selected.
        - `activation_fee_cents[c]` (from category table): fixed fee if any item in category `c` is selected.
        - `bonus_cents[p]` (from bundle table): bonus if both items in pair `p` are selected.
    - Constraint coefficients:
        - `usage[i, r]` (from usage table): amount of resource `r` used per unit of item `i`.
        - `capacity[r]` (from capacity_ledger table): total available amount of resource `r` (sum of opening and reservation entries).
        - `minimum_lot[i]`, `maximum_order[i]` (from item table): lower and upper bounds for item `i` quantity if selected.
        - `authorized[i]` (from item table): 1 if item `i` is eligible, 0 otherwise.
        - `minimum_quantity[c]`, `maximum_quantity[c]` (from category table): unconditional lower and upper bounds for total quantity in category `c`.
        - Incompatible pairs: item pairs that cannot both be selected (from incompatible table).
        - Requires pairs: item pairs where selection of one requires selection of the other (from requires table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * q[i]) minus (item_fee_cents[i] * z[i])
    - Minus sum over all categories: (activation_fee_cents[c] * y[c])
    - Plus sum over all bundle pairs: (bonus_cents[p] * b[p])
7.  **Formulate Constraints:**
    - **Item Authorization and Lot/Order Bounds:** For each item `i`, enforce:
        - If authorized[i] = 1: minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; z[i] ∈ {0,1}; q[i] ∈ {0,1,...,maximum_order[i]}.
        - If authorized[i] = 0: q[i] = 0; z[i] = 0.
    - **Resource Capacity:** For each resource `r`, sum over all items: sum_i (usage[i, r] * q[i]) ≤ capacity[r].
    - **Category Quantity Bounds:** For each category `c`, sum over all items in category: minimum_quantity[c] ≤ sum_{i in c} q[i] ≤ maximum_quantity[c].
    - **Category Activation:** For each category `c`, y[c] = 1 if any z[i] = 1 for i in c; y[c] = 0 otherwise. Enforce: z[i] ≤ y[c] for all i in c; y[c] ≤ sum_{i in c} z[i].
    - **Bundle Bonus Activation:** For each bundle pair p = (i_a, i_b): b[p] = 1 iff z[i_a] = 1 and z[i_b] = 1; enforce: b[p] ≤ z[i_a], b[p] ≤ z[i_b], b[p] ≥ z[i_a] + z[i_b] - 1.
    - **Incompatibility:** For each incompatible pair (i, j): z[i] + z[j] ≤ 1.
    - **Requires Dependencies:** For each requires pair (i, j): q[i] > 0 ⇒ q[j] > 0; enforce: q[i] ≤ maximum_order[i] * z[i], q[j] ≥ z[i]; z[i] ≤ z[j].
[Abstract Model Plan END]