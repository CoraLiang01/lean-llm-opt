**Index sets**

- \(I\) – set of all item‑references that appear in the *item* tables.  
- \(C\) – set of all category identifiers that appear in the *category* table.  
- \(R\) – set of all resource identifiers that appear in the *capacity_ledger* and *usage* tables.  
- \(B\) – set of all bundle identifiers that appear in the *bundle* table.  
- \(K\) – set of all incompatible pairs that appear in the *incompatible* table.  
- \(K_{\text{req}}\) – set of all requires pairs that appear in the *requires* table.  

**Parameters**

| Parameter | Definition | Source table & column |
|-----------|------------|-----------------------|
| \(p_i\) | per‑unit benefit (cents) of item \(i\) (sum of all *amount_cents* components for that item) | *benefit* – column *amount_cents* (grouped by *item_ref*) |
| \(a_i\) | item‑specific activation fee (cents) | *item_fee* – column *activation_fee_cents* (matched on *item_ref*) |
| \(c_i\) | category of item \(i\) | *item* – column *category* |
| \(u_i\) | maximum order quantity for item \(i\) | *item* – column *maximum_order* |
| \(l_i\) | minimum lot quantity for item \(i\) | *item* – column *minimum_lot* |
| \(s_i\) | section (resource) of item \(i\) | *item* – column *location_id* |
| \(f_i\) | authorization flag for item \(i\) (1 = authorized, 0 = unauthorized) | *item* – column *authorized* |
| \(b_{i,r}\) | usage of resource \(r\) per unit of item \(i\) (ml) | *usage* – column *amount* (unit already ml) |
| \(U_r\) | capacity of resource \(r\) (ml) | *capacity_ledger* – column *amount* |
| \(A_c\) | activation fee for category \(c\) (cents) | *category* – column *activation_fee_cents* |
| \(q^{\min}_c, q^{\max}_c\) | minimum and maximum total quantity for category \(c\) | *category* – columns *minimum_quantity*, *maximum_quantity* |
| \(b_{b}\) | bonus for bundle \(b\) (cents) | *bundle* – column *bonus_cents* |
| \((i_b^1,i_b^2)\) | item pair in bundle \(b\) | *bundle* – columns *item_a*, *item_b* |
| \((i_k^1,i_k^2)\) | incompatible item pair in \(K\) | *incompatible* – columns *item_a*, *item_b* |
| \((i_{req}^1,i_{req}^2)\) | requires pair in \(K_{\text{req}}\) | *requires* – columns *item_ref*, *prerequisite_ref* |

**Decision variables**

| Symbol | Type | Meaning |
|--------|------|---------|
| \(x_i\) | integer \(\ge 0\) | number of packs of item \(i\) to display |
| \(y_i\) | binary | 1 if item \(i\) is selected (positive quantity), 0 otherwise |
| \(z_c\) | binary | 1 if any item in category \(c\) is selected |
| \(w_b\) | binary | 1 if both items in bundle \(b\) are selected |

**Model**

\[
\begin{aligned}
\max\;&
\sum_{i\in I} p_i\,x_i
\;-\;
\sum_{i\in I} a_i\,y_i
\;-\;
\sum_{c\in C} A_c\,z_c
\;+\;
\sum_{b\in B} b_{b}\,w_b
\\[4pt]
\text{s.t.}\;&
\sum_{i\in I} b_{i,r}\,x_i \;\le\; U_r
\qquad \forall r\in R
\\[4pt]
& x_i \;\le\; u_i\,y_i
\qquad \forall i\in I
\\[4pt]
& x_i \;\ge\; l_i\,y_i
\qquad \forall i\in I
\\[4pt]
& y_i \;\le\; f_i
\qquad \forall i\in I
\\[4pt]
& z_c \;\ge\; y_i
\qquad \forall c\in C,\; i\in I:\; c_i=c
\\[4pt]
& \sum_{i\in I:\;c_i=c} x_i \;\ge\; q^{\min}_c\,z_c
\qquad \forall c\in C
\\[4pt]
& \sum_{i\in I:\;c_i=c} x_i \;\le\; q^{\max}_c\,z_c
\qquad \forall c\in C
\\[4pt]
& y_{i_k^1} + y_{i_k^2} \;\le\; 1
\qquad \forall (i_k^1,i_k^2)\in K
\\[4pt]
& y_{i_{req}^1} \;\le\; y_{i_{req}^2}
\qquad \forall (i_{req}^1,i_{req}^2)\in K_{\text{req}}
\\[4pt]
& w_b \;\le\; y_{i_b^1}
\qquad \forall b\in B
\\[4pt]
& w_b \;\le\; y_{i_b^2}
\qquad \forall b\in B
\\[4pt]
& w_b \;\ge\; y_{i_b^1} + y_{i_b^2} - 1
\qquad \forall b\in B
\\[4pt]
& x_i \in \mathbb{Z}_{\ge 0},\; y_i\in\{0,1\},\; z_c\in\{0,1\},\; w_b\in\{0,1\}
\qquad \forall i\in I,\; c\in C,\; b\in B
\end{aligned}
\]

**Data‑mapping notes**

- Every parameter is linked to a single source table and column as listed in the table above; no additional aggregation beyond the per‑item benefit sum is performed in the model.
- The *usage* and *capacity_ledger* amounts are in millilitres.
- Unauthorized items are forced to zero quantity by the constraint \(y_i \le f_i\) (with \(f_i=0\) for unauthorized items).
- The binary variables \(z_c\) and \(w_b\) ensure that category activation fees and bundle bonuses are counted at most once, regardless of how many items in the category or bundle are selected.
- All constraints are linear; the model is an integer linear program.