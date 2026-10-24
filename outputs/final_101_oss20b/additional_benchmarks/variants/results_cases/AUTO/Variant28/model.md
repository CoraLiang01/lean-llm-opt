**Mathematical Model**

Let  

* \(I\) – set of all items (rows of *item*).  
* \(C\) – set of all categories (rows of *category*).  
* \(R\) – set of all resources that appear in *usage* or *capacity_ledger*.  
* \(B\) – set of all bundles (rows of *bundle*).  
* \(P\) – set of all prerequisite links (rows of *requires*).  
* \(Q\) – set of all incompatibility pairs (rows of *incompatible*).

Parameters (all taken directly from the source tables)

| Parameter | Definition | Source |
|-----------|------------|--------|
| \(u_i\) | unit benefit (cents) of item \(i\) | *item*.\(unit_benefit_cents\) |
| \(f_i\) | item fee (cents) of item \(i\) | *item*.\(item_fee_cents\) |
| \(a_i\) | minimum lot of item \(i\) | *item*.\(minimum_lot\) |
| \(M_i\) | maximum order of item \(i\) | *item*.\(maximum_order\) |
| \(\alpha_i\) | binary indicator that item \(i\) is authorized (1) | *item*.\(authorized\) |
| \(c_{cr}\) | per‑unit usage of resource \(r\) by item \(i\) | *usage*.\(amount\) (converted to common unit) |
| \(s_{r}\) | signed capacity of resource \(r\) | sum of *capacity_ledger*.\(amount\) for that resource |
| \(A_c\) | activation fee (cents) of category \(c\) | *category*.\(activation_fee_cents\) |
| \(q^{\min}_c\) | minimum quantity of category \(c\) | *category*.\(minimum_quantity\) |
| \(q^{\max}_c\) | maximum quantity of category \(c\) | *category*.\(maximum_quantity\) |
| \(b_k\) | bonus (cents) of bundle \(k\) | *bundle*.\(bonus_cents\) |
| \(i(k,1), i(k,2)\) | the two items in bundle \(k\) | *bundle*.\(item_a\), *bundle*.\(item_b\) |
| \(p(i,j)\) | prerequisite link: item \(i\) requires item \(j\) | *requires*.\(item_ref\), *requires*.\(prerequisite_ref\) |
| \(q(i,j)\) | incompatibility link: items \(i\) and \(j\) cannot both be ordered | *incompatible*.\(item_a\), *incompatible*.\(item_b\) |

Decision variables  

* \(q_i \in \mathbb{Z}_{\ge 0}\) – quantity (cases) of item \(i\).  
* \(z_i \in \{0,1\}\) – 1 iff \(q_i>0\).  
* \(w_c \in \{0,1\}\) – 1 iff any item of category \(c\) is ordered.  
* \(b_k \in \{0,1\}\) – 1 iff both items of bundle \(k\) are ordered.

**Objective**

\[
\max \;\; \sum_{i\in I} \bigl(u_i\,q_i - f_i\,z_i\bigr)
          \;-\; \sum_{c\in C} A_c\,w_c
          \;+\; \sum_{k\in B} b_k\,b_k
\]

**Constraints**

1. **Item quantity bounds (authorized only)**  
   \[
   0 \le q_i \le M_i\,z_i,\qquad
   q_i \ge a_i\,z_i,\qquad
   z_i \le \alpha_i,\qquad
   z_i \in \{0,1\}
   \]

2. **Category limits**  
   \[
   q^{\min}_c\,w_c \;\le\; \sum_{i\in I:\,cat(i)=c} q_i
   \;\le\; q^{\max}_c\,w_c,\qquad
   w_c \in \{0,1\}
   \]
   \[
   w_c \;\ge\; z_i \quad \forall i\in I:\,cat(i)=c
   \]

3. **Incompatibility**  
   \[
   z_i + z_j \;\le\; 1 \quad \forall (i,j)\in Q
   \]

4. **Prerequisite**  
   \[
   z_i \;\le\; z_j \quad \forall (i,j)\in P
   \]

5. **Bundle bonus**  
   \[
   b_k \;\le\; z_{i(k,1)},\qquad
   b_k \;\le\; z_{i(k,2)},\qquad
   b_k \;\ge\; z_{i(k,1)} + z_{i(k,2)} - 1,\qquad
   b_k \in \{0,1\}
   \]

6. **Resource capacities**  
   \[
   \sum_{i\in I} c_{ir}\,q_i \;\le\; s_r \quad \forall r\in R
   \]

7. **Integrality**  
   \[
   q_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
   \]

---

### Data Mapping Rules

| Index | Source table | Key column(s) | Parameter mapping |
|-------|--------------|---------------|-------------------|
| \(i\) | *item* | *item_ref* | all item‑level parameters (u, f, a, M, α, category) |
| \(c\) | *category* | *category* | category parameters (A, qmin, qmax) |
| \(r\) | *usage* / *capacity_ledger* | *resource* | usage \(c_{ir}\) and capacity \(s_r\) |
| \(k\) | *bundle* | *table* (bundle) | bonus \(b_k\) and item links \(i(k,1), i(k,2)\) |
| \(p\) | *requires* | *item_ref*, *prerequisite_ref* | prerequisite links |
| \(q\) | *incompatible* | *item_a*, *item_b* | incompatibility links |

All numeric values are taken verbatim from the source rows; no additional aggregation or filtering is performed beyond the explicit joins shown above. Resource amounts and capacities use matching units (ml, wh, minute).