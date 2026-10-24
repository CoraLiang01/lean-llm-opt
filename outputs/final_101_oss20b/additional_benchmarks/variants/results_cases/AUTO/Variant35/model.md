**Mathematical model**

Let  

* \(I\) – set of all item references.  
* \(C\) – set of all categories.  
* \(R\) – set of all resources (PC, CONSOLE, MOBILE).  
* \(B\) – set of all bundles.  
* \(K\) – set of all incompatibility pairs.  
* \(L\) – set of all requires pairs.

Decision variables  

\[
\begin{aligned}
q_i &\in \mathbb{Z}_{+} &&\text{units of item }i\\
z_i &\in \{0,1\} &&\text{indicator that }q_i>0\\
w_c &\in \{0,1\} &&\text{indicator that category }c\text{ is used}\\
b_k &\in \{0,1\} &&\text{indicator that bundle }k\text{ is activated}
\end{aligned}
\qquad i\in I,\;c\in C,\;k\in B
\]

Parameters (all taken directly from the supplied tables)

| Symbol | Meaning | Source table & column |
|--------|---------|-----------------------|
| \(b_i\) | benefit per unit (USD cents) | `benefit` – amount, currency → convert with `fx` |
| \(u_{i,r}\) | usage per unit of item \(i\) on resource \(r\) | `usage` – amount (MB) |
| \(\text{cap}_r\) | total capacity of resource \(r\) | `capacity_ledger` – amount (MB) |
| \(\text{fee}_i\) | item‑fee (USD cents) | `item_fee` – activation_fee_cents |
| \(\text{minlot}_i,\text{maxord}_i\) | min‑lot, max‑order | `item` – minimum_lot, maximum_order |
| \(\text{auth}_i\) | 1 if authorized, 0 otherwise | `item` – authorized |
| \(\text{cat}_i\) | category of item \(i\) | `item` – category |
| \(\text{minq}_c,\text{maxq}_c\) | category quantity bounds | `category` – minimum_quantity, maximum_quantity |
| \(\text{actfee}_c\) | category activation fee | `category` – activation_fee_cents |
| \(\text{bonus}_k\) | bundle bonus (USD cents) | `bundle` – bonus_cents |
| \((a_k,b_k)\) | items in bundle \(k\) | `bundle` – item_a, item_b |
| \((i,j)\in K\) | incompatible pair | `incompatible` – item_a, item_b |
| \((i,j)\in L\) | requires pair | `requires` – item_ref, prerequisite_ref |

**Objective**

\[
\max \;\; \sum_{i\in I} b_i\,q_i
      \;-\;\sum_{i\in I} \text{fee}_i\,z_i
      \;-\;\sum_{c\in C} \text{actfee}_c\,w_c
      \;+\;\sum_{k\in B} \text{bonus}_k\,b_k
\]

**Constraints**

1. **Quantity bounds (authorized items)**  
   \[
   \text{auth}_i\,\text{minlot}_i \;\le\; q_i \;\le\; \text{auth}_i\,\text{maxord}_i
   \qquad \forall i\in I
   \]

2. **Activation indicator**  
   \[
   q_i \;\le\; M\,z_i \qquad \forall i\in I
   \]
   (with \(M\) a sufficiently large constant, e.g. \(\max_i\text{maxord}_i\))

3. **Category activation**  
   \[
   \sum_{i:\text{cat}_i=c} q_i \;\ge\; \text{minq}_c\,w_c
   \quad
   \sum_{i:\text{cat}_i=c} q_i \;\le\; \text{maxq}_c\,w_c
   \qquad \forall c\in C
   \]

4. **Resource capacity**  
   \[
   \sum_{i\in I} u_{i,r}\,q_i \;\le\; \text{cap}_r
   \qquad \forall r\in R
   \]

5. **Incompatibility**  
   \[
   z_i + z_j \;\le\; 1 \qquad \forall (i,j)\in K
   \]

6. **Requires**  
   \[
   z_i \;\le\; z_j \qquad \forall (i,j)\in L
   \]

7. **Bundle activation**  
   \[
   b_k \;\le\; z_{a_k}, \quad
   b_k \;\le\; z_{b_k}, \quad
   b_k \;\ge\; z_{a_k}+z_{b_k}-1
   \qquad \forall k\in B
   \]

8. **Variable domains**  
   \[
   q_i \in \mathbb{Z}_{+},\;
   z_i,w_c,b_k \in \{0,1\}
   \]

---

### Data‑Mapping rules

| Parameter | Source table | Source column | Notes |
|-----------|--------------|---------------|-------|
| \(b_i\) | `benefit` | amount, currency | Convert using `fx` (usd_cents_numerator/denominator) |
| \(u_{i,r}\) | `usage` | amount (MB) | MB |
| \(\text{cap}_r\) | `capacity_ledger` | amount (MB) | |
| \(\text{fee}_i\) | `item_fee` | activation_fee_cents | |
| \(\text{minlot}_i,\text{maxord}_i\) | `item` | minimum_lot, maximum_order | |
| \(\text{auth}_i\) | `item` | authorized | |
| \(\text{cat}_i\) | `item` | category | |
| \(\text{minq}_c,\text{maxq}_c\) | `category` | minimum_quantity, maximum_quantity | |
| \(\text{actfee}_c\) | `category` | activation_fee_cents | |
| \(\text{bonus}_k\) | `bundle` | bonus_cents | |
| \((a_k,b_k)\) | `bundle` | item_a, item_b | |
| \((i,j)\in K\) | `incompatible` | item_a, item_b | |
| \((i,j)\in L\) | `requires` | item_ref, prerequisite_ref | |

All parameters are taken **exactly** from the corresponding rows of the supplied CSV files; no additional aggregation or filtering is performed beyond the explicit conversions described. The model above is a complete, compact symbolic formulation that maximizes the net licensing benefit in USD cents while respecting all capacity, diversification, and contractual constraints.