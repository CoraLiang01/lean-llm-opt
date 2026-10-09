[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer quantities of authorized air-conditioner models to place in each storage area at FC_EAST_HVAC, maximizing net value (benefit minus all fixed and variable fees, plus bonuses), subject to area volume limits, category quantity bounds, incompatibility and prerequisite rules, and bundle bonuses, using only the supplied item rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, group activation, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: all item_ref values from the supplied item tables (authorized items only).
    - Categories: all category values from the category table.
    - Resources: all resource values from the capacity_ledger and usage tables.
    - Bundles: all (item_a, item_b) pairs from the bundle table.
    - Incompatibles: all (item_a, item_b) pairs from the incompatible table.
    - Prerequisites: all (item_ref, prerequisite_ref) pairs from the requires table.
4.  **Define Decision Variables:**
    - `q[i]` = integer quantity of item i to place (i in Items). Type: GRB.INTEGER, domain: 0 or [minimum_lot[i], maximum_order[i]] if selected, 0 if not selected.
    - `z[i]` = 1 if item i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `g[c]` = 1 if any item in category c is selected, 0 otherwise. Type: GRB.BINARY.
    - `b[a,b]` = 1 if both items a and b in bundle (a,b) are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - unit_benefit_cents[i] (from item tables): per-unit benefit for item i.
        - item_fee_cents[i] (from item tables): fixed fee if any of item i is selected.
        - activation_fee_cents[c] (from category table): fixed fee if any item in category c is selected.
        - bonus_cents[a,b] (from bundle table): bonus if both items a and b are selected.
    - Constraint coefficients:
        - usage[i,r] (from usage tables): per-unit resource usage of item i for resource r.
    - Constraint RHS:
        - capacity[r] (sum of capacity_ledger amounts for resource r): total available for each resource.
        - minimum_quantity[c], maximum_quantity[c] (from category table): lower and upper bounds for total quantity in each category.
        - minimum_lot[i], maximum_order[i] (from item tables): lower and upper bounds for item i if selected.
        - authorized[i] (from item tables): 1 if item i is eligible, 0 otherwise.
6.  **Formulate Objective:** Maximize total net benefit in cents:
        sum over i of (unit_benefit_cents[i] * q[i] - item_fee_cents[i] * z[i])
      minus
        sum over c of (activation_fee_cents[c] * g[c])
      plus
        sum over (a,b) in Bundles of (bonus_cents[a,b] * b[a,b])
7.  **Formulate Constraints:**
    - Resource Capacity: For each resource r, sum over i of (usage[i,r] * q[i]) ≤ capacity[r].
    - Item Authorization: For each item i, q[i] = 0 if authorized[i] = 0.
    - Item Quantity Bounds: For each item i, minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; q[i] = 0 if z[i] = 0.
    - Category Quantity Bounds: For each category c, minimum_quantity[c] ≤ sum over i in c of q[i] ≤ maximum_quantity[c].
    - Category Activation: For each category c, for all i in c, z[i] ≤ g[c]; and g[c] ≤ sum over i in c of z[i].
    - Incompatibility: For each incompatible pair (a,b), z[a] + z[b] ≤ 1.
    - Prerequisite: For each (i, prereq) in requires, z[i] ≤ z[prereq].
    - Bundle Activation: For each bundle (a,b), b[a,b] ≤ z[a], b[a,b] ≤ z[b], and b[a,b] ≥ z[a] + z[b] - 1.
[Abstract Model Plan END]