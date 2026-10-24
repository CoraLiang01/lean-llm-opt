**Index sets**

- \(I\) – set of all item references that appear in the *item* table.  
- \(C\) – set of all category identifiers that appear in the *category* table.  
- \(R\) – set of all resource identifiers that appear in the *usage* or *capacity_ledger* tables.  
- \(B\) – set of all bundle identifiers that appear in the *bundle* table.  
- \(K\) – set of all prerequisite pairs that appear in the *requires* table.  
- \(P\) – set of all incompatible pairs that appear in the *incompatible* table.  

**Parameters (all taken directly from the source tables)**  

| Symbol | Definition | Source table / column |
|--------|------------|-----------------------|
| \(a_i\) | unit benefit (cents) of item \(i\) | *item.unit_benefit_cents* |
| \(f_i\) | item fee (cents) of item \(i\) | *item.item_fee_cents* |
| \(m_i\) | minimum lot of item \(i\) | *item.minimum_lot* |
| \(M_i\) | maximum order of item \(i\) | *item.maximum_order* |
| \(auth_i\) | 1 if item \(i\) is authorized, 0 otherwise | *item.authorized* |
| \(c_{c}^{min}\) | minimum quantity for category \(c\) | *category.minimum_quantity* |
| \(c_{c}^{max}\) | maximum quantity for category \(c\) | *category.maximum_quantity* |
| \(F_c\) | activation fee (cents) for category \(c\) | *category.activation_fee_cents* |
| \(u_{i,r}\) | usage of resource \(r\) per unit of item \(i\) | *usage.amount* (joined on *usage.item_ref* and *usage.resource*) |
| \(L_{r}\) | signed capacity‑ledger amount for resource \(r\) | *capacity_ledger.amount* (joined on *capacity_ledger.resource*) |
| \(b_k\) | bonus (cents) for bundle \(k\) | *bundle.bonus_cents* |
| \(i_k^a,i_k^b\) | item references in bundle \(k\) | *bundle.item_a*, *bundle.item_b* |
| \((i,p)\) | prerequisite pair (item, prerequisite) | *requires.item_ref*, *requires.prerequisite_ref* |
| \((i,j)\) | incompatible pair (item, item) | *incompatible.item_a*, *incompatible.item_b* |

All numeric parameters are in the same unit (cents for monetary values, ml for resources).

**Decision variables**

| Symbol | Type | Meaning |
|--------|------|---------|
| \(q_i\) | integer \(\ge 0\) | quantity of item \(i\) to order |
| \(z_i\) | binary | 1 if item \(i\) is selected (i.e., \(q_i>0\)), 0 otherwise |
| \(w_c\) | binary | 1 if category \(c\) is used (i.e., any item in \(c\) is selected) |
| \(b_k\) | binary | 1 if both items of bundle \(k\) are selected |

**Model**

\[
\begin{aligned}
\max\;&
\sum_{i\in I} a_i\,q_i
-\sum_{i\in I} f_i\,z_i
-\sum_{c\in C} F_c\,w_c
+\sum_{k\in B} b_k\,b_k
\\[4pt]
\text{s.t.}\;&
\text{(1) Item bounds} \\
& m_i\,z_i \;\le\; q_i \;\le\; M_i\,z_i \qquad \forall i\in I \\
& z_i \;\le\; auth_i \qquad \forall i\in I \\[4pt]
& \text{(2) Category limits} \\
& \sum_{i\in I_c} q_i \;\ge\; c_{c}^{min}\,w_c \qquad \forall c\in C \\
& \sum_{i\in I_c} q_i \;\le\; c_{c}^{max}\,w_c \qquad \forall c\in C \\
& w_c \;\ge\; z_i \qquad \forall c\in C,\; i\in I_c \\[4pt]
& \text{(3) Resource capacity} \\
& \sum_{i\in I} u_{i,r}\,q_i \;\le\; L_{r} \qquad \forall r\in R \\[4pt]
& \text{(4) Incompatibility} \\
& z_{i} + z_{j} \;\le\; 1 \qquad \forall (i,j)\in P \\[4pt]
& \text{(5) Requires} \\
& z_{i} \;\le\; z_{p} \qquad \forall (i,p)\in K \\[4pt]
& \text{(6) Bundle bonus} \\
& b_k \;\le\; z_{i_k^a} \qquad \forall k\in B \\
& b_k \;\le\; z_{i_k^b} \qquad \forall k\in B \\
& b_k \;\ge\; z_{i_k^a} + z_{i_k^b} - 1 \qquad \forall k\in B \\[4pt]
& \text{(7) Variable domains} \\
& q_i \in \mathbb{Z}_{\ge 0} \qquad \forall i\in I \\
& z_i,w_c,b_k \in \{0,1\} \qquad \forall i\in I,\; c\in C,\; k\in B
\end{aligned}
\]

**Data‑mapping rules**

- Every parameter value is taken from the corresponding column of the source table; no aggregation or transformation is performed beyond the unit consistency already ensured by the *unit_preparation* step.
- The set \(I\) consists of all rows in *item* where *item.authorized* is 1 or 0; rows with *authorized* = 0 are still present but forced to \(z_i=0\) by the constraint \(z_i \le auth_i\).
- The set \(C\) is the distinct list of *category* identifiers in the *category* table.
- The set \(R\) is the distinct list of *resource* identifiers that appear in either *usage* or *capacity_ledger*.
- The set \(B\) is the list of all rows in *bundle*; each row supplies the two item references and the bonus amount.
- The set \(K\) is the list of all rows in *requires*; each row supplies a pair (item, prerequisite).
- The set \(P\) is the list of all rows in *incompatible*; each row supplies a pair (item, item).
- All binary variables \(z_i,w_c,b_k\) are defined only for indices that appear in the corresponding sets; no additional indices are introduced.
- The capacity‑ledger amounts \(L_r\) are summed over all ledger entries for resource \(r\); the resulting signed total is used directly in the resource capacity constraints.

This formulation uses only the data returned by the query, preserves all specified limits and fees, and maximizes the net benefit in USD cents.