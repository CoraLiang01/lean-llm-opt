## Mathematical Model

Let  

* \(I\) – set of all item references (from **item** table)  
* \(C\) – set of all categories (from **category** table)  
* \(B\) – set of all bundles (from **bundle** table)  
* \(R\) – set of all resources (from **usage** and **capacity_ledger** tables)  

Parameters (all in the units shown in the source tables)

| Parameter | Symbol | Source table & column | Description |
|-----------|--------|-----------------------|-------------|
| unit benefit per unit | \(p_i\) | `item.unit_benefit_cents` | revenue per unit of item \(i\) |
| item fee (once) | \(f_i\) | `item.item_fee_cents` | fixed fee if item \(i\) is used |
| minimum lot | \(l_i\) | `item.minimum_lot` | lower bound on quantity if item \(i\) is used |
| maximum order | \(u_i\) | `item.maximum_order` | upper bound on quantity |
| authorization flag | \(a_i\in\{0,1\}\) | `item.authorized` | 1 if item is allowed, 0 otherwise |
| item category | \(c(i)\in C\) | `item.category` | category of item \(i\) |
| activation fee | \(F_c\) | `category.activation_fee_cents` | fee if any item in category \(c\) is used |
| category min quantity | \(L_c\) | `category.minimum_quantity` | lower bound on total units in category \(c\) |
| category max quantity | \(U_c\) | `category.maximum_quantity` | upper bound on total units in category \(c\) |
| bundle bonus | \(B_k\) | `bundle.bonus_cents` | bonus if both items in bundle \(k\) are used |
| bundle items | \(a_k,b_k\in I\) | `bundle.item_a`, `bundle.item_b` | items in bundle \(k\) |
| prerequisite | \(pr_k\in I\) | `requires.prerequisite_ref` | prerequisite item for item in `requires.item_ref` |
| usage coefficient | \(u_{i,r}\) | `usage.amount` | units of resource \(r\) consumed per unit of item \(i\) |
| capacity limit | \(C_r\) | sum of `capacity_ledger.amount` for resource \(r\) | total available amount of resource \(r\) |

Decision variables  

* \(q_i \in \mathbb{Z}_{\ge 0}\) – quantity of item \(i\)  
* \(z_i \in \{0,1\}\) – 1 if item \(i\) is used (i.e., \(q_i>0\))  
* \(w_c \in \{0,1\}\) – 1 if category \(c\) is activated (any item in \(c\) used)  
* \(b_k \in \{0,1\}\) – 1 if both items of bundle \(k\) are used  

### Objective  

\[
\max \;\; \sum_{i\in I} p_i\,q_i \;-\; \sum_{i\in I} f_i\,z_i \;-\; \sum_{c\in C} F_c\,w_c \;+\; \sum_{k\in B} B_k\,b_k
\]

### Constraints  

1. **Quantity bounds (authorized items only)**  
\[
l_i\,z_i \;\le\; q_i \;\le\; u_i\,z_i \qquad \forall i\in I
\]
\[
z_i \;\le\; a_i \qquad \forall i\in I
\]

2. **Category activation**  
\[
w_c \;\ge\; z_i \qquad \forall i\in I,\; c=c(i)
\]
\[
w_c \;\le\; \sum_{i:c(i)=c} z_i \qquad \forall c\in C
\]

3. **Category quantity limits**  
\[
L_c\,w_c \;\le\; \sum_{i:c(i)=c} q_i \;\le\; U_c\,w_c \qquad \forall c\in C
\]

4. **Incompatibility**  
\[
z_{a} + z_{b} \;\le\; 1 \qquad \forall (a,b)\in \text{incompatible}
\]

5. **Prerequisite**  
\[
z_{\text{item_ref}} \;\le\; z_{\text{prerequisite_ref}} \qquad \forall (\text{item_ref},\text{prerequisite_ref})\in \text{requires}
\]

6. **Bundle bonus**  
\[
b_k \;\le\; z_{a_k}, \quad b_k \;\le\; z_{b_k}, \quad b_k \;\ge\; z_{a_k}+z_{b_k}-1 \qquad \forall k\in B
\]

7. **Resource capacity**  
\[
\sum_{i\in I} u_{i,r}\,q_i \;\le\; C_r \qquad \forall r\in R
\]

8. **Variable domains**  
\[
q_i \in \mathbb{Z}_{\ge 0},\; z_i,w_c,b_k \in \{0,1\}
\]

---

## Data Mapping Rules

| Index | Source tables | Key columns | Parameter mapping |
|-------|---------------|-------------|-------------------|
| \(I\) | `item` (file_5_view_0, file_6_view_0) | `item.item_ref` | all items |
| \(C\) | `category` (file_2_view_0) | `category.category` | all categories |
| \(B\) | `bundle` (file_0_view_0) | `bundle.item_a`, `bundle.item_b` | bundle pairs |
| \(R\) | `usage` (file_9_view_0, file_10_view_0) and `capacity_ledger` (file_1_view_0) | `usage.resource`, `capacity_ledger.resource` | all resources |
| Parameters | see table above | columns as listed | direct assignment |
| Capacity limits \(C_r\) | `capacity_ledger.amount` | sum over all entries with same `resource` | aggregated per resource |
| Usage coefficients \(u_{i,r}\) | `usage.amount` | map by `item_ref` and `resource` | direct assignment |

All numeric values are taken exactly as they appear in the source rows; no additional conversion is performed beyond the unit preparation already applied (all amounts are in ml). The model uses the full set of records returned by the query; no records are omitted or duplicated.