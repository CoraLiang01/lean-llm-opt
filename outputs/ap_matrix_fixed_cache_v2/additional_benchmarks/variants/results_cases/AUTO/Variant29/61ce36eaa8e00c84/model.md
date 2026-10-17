[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category activation fees, bundle bonuses, and subject to storage, staff time, and energy resource limits, category-level minimum and maximum total quantities, item-level minimum and maximum order sizes, incompatibility and prerequisite requirements, and bundle bonus eligibility.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items: all item_ref values from the 'item' table (export_03.csv) with authorized=1.
    - Categories: all category values from the 'category' table (export_05.csv).
    - Resources: all resource values from the 'usage' and 'capacity_ledger' tables (export_04.csv, export_09.csv).
    - Incompatible pairs: all (item_a, item_b) from the 'incompatible' table (export_06.csv).
    - Prerequisite pairs: all (item_ref, prerequisite_ref) from the 'requires' table (export_01.csv).
    - Bundles: all (item_a, item_b) from the 'bundle' table (export_08.csv).
4.  **Define Decision Variables:**
    -   `q[i]` = integer quantity ordered of item i (i in Items). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item i is ordered (q[i] ≥ minimum_lot[i]), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered (∃i in c: y[i]=1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle (item_a, item_b) are ordered (y[item_a]=y[item_b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   unit_benefit_cents[i] (from 'item' table, export_03.csv)
        -   item_fee_cents[i] (from 'item' table, export_03.csv)
        -   activation_fee_cents[c] (from 'category' table, export_05.csv)
        -   bonus_cents[bundle] (from 'bundle' table, export_08.csv)
    -   Constraint coefficients:
        -   amount[i,r] (per-unit usage of resource r by item i, from 'usage' table, export_04.csv)
    -   Constraint RHS:
        -   For each resource r: sum of amount over all entries for r in 'capacity_ledger' table (export_09.csv), after converting all units to a common base (ml for space, minute for labor, wh for power).
        -   minimum_quantity[c], maximum_quantity[c] (from 'category' table, export_05.csv)
        -   minimum_lot[i], maximum_order[i] (from 'item' table, export_03.csv)
    -   Logical constraints:
        -   Incompatible pairs (from 'incompatible' table, export_06.csv)
        -   Prerequisite pairs (from 'requires' table, export_01.csv)
        -   Authorization: only items with authorized=1 (from 'item' table, export_03.csv) are eligible for selection and for triggering bonuses.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        sum over i of (unit_benefit_cents[i] * q[i] - item_fee_cents[i] * y[i])
      minus sum over c of (activation_fee_cents[c] * z[c])
      plus sum over all eligible bundles of (bonus_cents[bundle] * b[bundle]).
      (All terms in cents; only authorized items can be selected or trigger bonuses.)
7.  **Formulate Constraints:**
    -   Resource constraints: For each resource r,
            sum over i of (amount[i,r] * q[i] * unit_conversion[i,r]) ≤ total_capacity[r]
        where unit_conversion[i,r] converts all usage and capacity to a common unit (e.g., kwh→wh, liter→ml, hour→minute).
    -   Item order bounds: For each item i,
            q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i]
        Enforced via: minimum_lot[i] * y[i] ≤ q[i] ≤ maximum_order[i] * y[i], q[i] ≥ 0 integer, y[i] ∈ {0,1}.
    -   Category quantity bounds: For each category c,
            minimum_quantity[c] ≤ sum over i in c of q[i] ≤ maximum_quantity[c]
    -   Category activation: For each category c,
            For all i in c: y[i] ≤ z[c]; z[c] ≤ sum over i in c of y[i]
    -   Incompatibility: For each incompatible pair (i,j),
            y[i] + y[j] ≤ 1
    -   Prerequisite: For each (i, prereq) in requires,
            y[i] ≤ y[prereq]
    -   Bundle bonuses: For each bundle (i,j),
            b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1
        and b[bundle]=0 if either i or j is unauthorized.
[Abstract Model Plan END]