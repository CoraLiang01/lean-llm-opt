**Mathematical Model**

Let  

* \(I\) – set of all items (configuration IDs)  
* \(C\) – set of all categories  
* \(R\) – set of all resources  
* \(B\) – set of all bundles  

Decision variables  

\[
\begin{aligned}
q_i &\in \mathbb{Z}_{\ge 0} &&\text{quantity of item }i\\
z_i &\in \{0,1\} &&\text{indicator that }q_i>0\\
y_c &\in \{0,1\} &&\text{indicator that category }c\text{ is used}\\
b_k &\in \{0,1\} &&\text{indicator that bundle }k\text{ bonus is earned}
\end{aligned}
\]

Parameters (all in USD cents unless otherwise noted)

| Symbol | Meaning | Source table & column |
|--------|---------|----------------------|
| \(p_i\) | unit benefit of item \(i\) | `item` – `unit_benefit_cents` |
| \(f_i\) | item fee of item \(i\) | `item` – `item_fee_cents` |
| \(L_i\) | minimum lot of item \(i\) | `item` – `minimum_lot` |
| \(U_i\) | maximum order of item \(i\) | `item` – `maximum_order` |
| \(a_i\) | category of item \(i\) | `item` – `category` |
| \(c_{\min c}\) | minimum quantity for category \(c\) | `category` – `minimum_quantity` |
| \(c_{\max c}\) | maximum quantity for category \(c\) | `category` – `maximum_quantity` |
| \(F_c\) | activation fee for category \(c\) | `category` – `activation_fee_cents` |
| \(u_{ir}\) | usage of resource \(r\) per unit of item \(i\) | `usage` – `amount` (unit matches `capacity_ledger`) |
| \(A_r\) | capacity ledger amount for resource \(r\) | `capacity_ledger` – `amount` |
| \(b_{k}\) | bonus for bundle \(k\) | `bundle` – `bonus_cents` |
| \(a_k,\,b_k\) | items in bundle \(k\) | `bundle` – `item_a`, `item_b` |
| \(r_{ij}\) | prerequisite: item \(i\) requires item \(j\) | `requires` – `item_ref`, `prerequisite_ref` |
| \(s_{ab}\) | incompatibility: items \(a\) and \(b\) cannot both be ordered | `incompatible` – `item_a`, `item_b` |

Big‑\(M\) for each category: \(M_c = c_{\max c}\).

---

### Objective

\[
\max \;\; \sum_{i\in I} p_i\,q_i \;-\; \sum_{i\in I} f_i\,z_i \;-\; \sum_{c\in C} F_c\,y_c \;+\; \sum_{k\in B} b_k\,b_k
\]

---

### Constraints

1. **Item quantity bounds (authorized only)**  

   \[
   L_i\,z_i \;\le\; q_i \;\le\; U_i\,z_i \qquad \forall i\in I
   \]

   (For unauthorized items \(U_i=0,\;L_i=0\), forcing \(q_i=0\).)

2. **Category quantity limits**  

   \[
   c_{\min c}\;\le\; \sum_{i:a_i=c} q_i \;\le\; c_{\max c} \qquad \forall c\in C
   \]

3. **Category activation fee**  

   \[
   \sum_{i:a_i=c} q_i \;\le\; M_c\,y_c \qquad \forall c\in C
   \]
   \[
   \sum_{i:a_i=c} q_i \;\ge\; y_c \qquad \forall c\in C
   \]

4. **Bundle bonus**  

   \[
   b_k \;\le\; z_{a_k}, \quad b_k \;\le\; z_{b_k}, \quad
   b_k \;\ge\; z_{a_k}+z_{b_k}-1 \qquad \forall k\in B
   \]

5. **Requires dependencies**  

   \[
   z_i \;\le\; z_j \qquad \forall (i\!\to\!j)\in \text{requires}
   \]

6. **Incompatible pairs**  

   \[
   z_a + z_b \;\le\; 1 \qquad \forall (a,b)\in \text{incompatible}
   \]

7. **Resource capacity limits**  

   \[
   \sum_{i\in I} u_{ir}\,q_i \;\le\; A_r \qquad \forall r\in R
   \]

8. **Variable domains**  

   \[
   q_i \in \mathbb{Z}_{\ge 0},\; z_i,y_c,b_k \in \{0,1\}
   \]

---

**Data Mapping**

| Table ID | Columns used | Notes |
|----------|--------------|-------|
| `file_5_view_0` (`item`) | `item_ref`, `category`, `authorized`, `minimum_lot`, `maximum_order`, `unit_benefit_cents`, `item_fee_cents` | All items; unauthorized items have `authorized=0` and are forced to zero by bounds. |
| `file_2_view_0` (`category`) | `category`, `minimum_quantity`, `maximum_quantity`, `activation_fee_cents` | Category limits and activation fees. |
| `file_8_view_0` (`usage`) | `item_ref`, `resource`, `amount` | Per‑unit resource consumption; units match those in `capacity_ledger`. |
| `file_1_view_0` (`capacity_ledger`) | `resource`, `amount` | Total available capacity per resource. |
| `file_0_view_0` (`bundle`) | `item_a`, `item_b`, `bonus_cents` | Bundle bonus definitions. |
| `file_7_view_0` (`requires`) | `item_ref`, `prerequisite_ref` | Requires relationships. |
| `file_4_view_0` (`incompatible`) | `item_a`, `item_b` | Incompatible pairs. |

All parameters are taken directly from the corresponding rows; no aggregation or transformation is performed beyond the unit consistency already ensured by the literal‑query‑unit‑view. The model above is a complete, compact symbolic formulation that respects every restriction stated in the query.