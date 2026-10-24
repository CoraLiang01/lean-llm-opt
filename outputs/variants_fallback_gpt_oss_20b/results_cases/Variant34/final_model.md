**Mathematical model**

Let  

* \(I\) – set of all item‑refs that appear in the *item* table.  
* \(C\) – set of all categories that appear in the *category* table.  
* \(R\) – set of all resources that appear in the *capacity_ledger* table.  
* \(B\) – set of all bundle rows.  
* \(K\) – set of all incompatible pairs.  
* \(L\) – set of all requires pairs.  

Decision variables  

\[
\begin{aligned}
x_i &\in \mathbb{Z}_{\ge 0} &&\text{packs of item }i\\
y_i &\in \{0,1\} &&\text{item }i\text{ is used}\\
z_c &\in \{0,1\} &&\text{category }c\text{ is used}\\
w_b &\in \{0,1\} &&\text{bundle }b\text{ is activated}
\end{aligned}
\qquad i\in I,\;c\in C,\;b\in B
\]

Parameters (all in the units shown in the source tables)

* \(B_i\) – per‑unit benefit (cents) of item \(i\)  
  \(\displaystyle B_i=\sum_{\text{rows }r:\,r.\text{item_ref}=i} r.\text{amount_cents}\)

* \(U_{i,r}\) – usage of resource \(r\) (ml) per pack of item \(i\)  
  \(\displaystyle U_{i,r}=\sum_{\text{rows }u:\,u.\text{item_ref}=i,\;u.\text{resource}=r} u.\text{amount}\)

* \(C_r\) – capacity of resource \(r\) (ml) from *capacity_ledger*  

* \(F_i\) – item fee (cents) from *item_fee*  

* \(\text{min}_c,\text{max}_c\) – lower/upper quantity limits for category \(c\) from *category*  

* \(A_c\) – activation fee (cents) for category \(c\) from *category*  

* \(\text{bonus}_b\) – bonus (cents) for bundle \(b\) from *bundle*  

* \(\text{minLot}_i,\text{maxOrder}_i\) – minimum lot and maximum order for item \(i\) from *item*  

* \(\text{auth}_i\in\{0,1\}\) – authorization flag from *item*  

* \(\text{cat}(i)\) – category of item \(i\) from *item*  

* \(\text{res}(i)\) – resource (section) of item \(i\) from *item*  

* \((i,j)\in K\) – incompatible pair from *incompatible*  

* \((i,j)\in L\) – requires pair from *requires*  

---

**Objective**

\[
\max \;\; \sum_{i\in I} B_i\,x_i \;+\; \sum_{b\in B} \text{bonus}_b\,w_b
\;-\; \sum_{i\in I} F_i\,y_i \;-\; \sum_{c\in C} A_c\,z_c
\]

---

**Constraints**

1. **Authorization**  
   \[
   y_i \le \text{auth}_i \qquad \forall i\in I
   \]

2. **Lot / order bounds**  
   \[
   \text{minLot}_i\,y_i \;\le\; x_i \;\le\; \text{maxOrder}_i\,y_i
   \qquad \forall i\in I
   \]

3. **Resource capacities**  
   \[
   \sum_{i\in I} U_{i,r}\,x_i \;\le\; C_r \qquad \forall r\in R
   \]

4. **Category quantity limits**  
   \[
   \text{min}_c\,z_c \;\le\; \sum_{i:\,\text{cat}(i)=c} x_i
   \;\le\; \text{max}_c\,z_c \qquad \forall c\in C
   \]

5. **Category activation**  
   \[
   z_c \;\ge\; y_i \quad \forall i:\,\text{cat}(i)=c
   \]
   \[
   z_c \;\le\; \sum_{i:\,\text{cat}(i)=c} y_i \qquad \forall c\in C
   \]

6. **Item fee activation**  
   \[
   y_i \;\le\; 1 \quad \text{(already binary)} \qquad \forall i\in I
   \]

7. **Incompatibility**  
   \[
   y_i + y_j \;\le\; 1 \qquad \forall (i,j)\in K
   \]

8. **Requires**  
   \[
   y_i \;\le\; y_j \qquad \forall (i,j)\in L
   \]

9. **Bundle activation**  
   \[
   w_b \;\le\; y_{i(b)} \quad
   w_b \;\le\; y_{j(b)} \qquad \forall b\in B
   \]
   \[
   w_b \;\ge\; y_{i(b)} + y_{j(b)} - 1 \qquad \forall b\in B
   \]

10. **Variable domains**  
    \[
    x_i \in \mathbb{Z}_{\ge 0},\; y_i,z_c,w_b \in \{0,1\}
    \qquad \forall i\in I,\;c\in C,\;b\in B
    \]

---

**Data‑mapping rules**

| Parameter | Source table | Source column(s) | Notes |
|-----------|--------------|------------------|-------|
| \(B_i\) | *benefit* | `item_ref`, `amount_cents` | Sum over all rows with same `item_ref`. |
| \(U_{i,r}\) | *usage* | `item_ref`, `resource`, `amount` | Sum over all rows with same `item_ref` and `resource`. |
| \(C_r\) | *capacity_ledger* | `resource`, `amount` | One row per resource. |
| \(F_i\) | *item_fee* | `item_ref`, `activation_fee_cents` | One row per item. |
| \(\text{min}_c,\text{max}_c\) | *category* | `category`, `minimum_quantity`, `maximum_quantity` | One row per category. |
| \(A_c\) | *category* | `category`, `activation_fee_cents` | One row per category. |
| \(\text{bonus}_b\) | *bundle* | `bonus_cents` | One row per bundle. |
| \(\text{minLot}_i,\text{maxOrder}_i\) | *item* | `item_ref`, `minimum_lot`, `maximum_order` | One row per item. |
| \(\text{auth}_i\) | *item* | `authorized` | 1 = authorized, 0 = not. |
| \(\text{cat}(i)\) | *item* | `category` | One row per item. |
| \(\text{res}(i)\) | *item* | `location_id` | One row per item. |
| \((i,j)\in K\) | *incompatible* | `item_a`, `item_b` | One row per incompatible pair. |
| \((i,j)\in L\) | *requires* | `item_ref`, `prerequisite_ref` | One row per requires pair. |

All numeric values are taken directly from the source tables; no additional aggregation or conversion is performed beyond the unit equivalence already applied in the *usage* and *capacity_ledger* tables (ml). The model is a mixed‑integer linear program that maximizes net merchandising benefit in cents.