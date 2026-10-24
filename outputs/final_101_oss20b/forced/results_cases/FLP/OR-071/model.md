**Symbolic Model**

Let  

* \(P\) – set of all product names in *41.csv*  
* \(x_p\) – production quantity of product \(p\in P\) (continuous, \(x_p\ge 0\))

Parameters (obtained from *41.csv*)  

* \(L_p\) – labor required per unit of product \(p\)  
* \(M_p\) – material required per unit of product \(p\)  
* \(S_p\) – selling price per unit of product \(p\)  
* \(V_p\) – variable cost per unit of product \(p\)

Fixed weekly operating cost: \(F = 4500\).

---

**Objective**

\[
\max_{x}\;\;\sum_{p\in P}\bigl(S_p-V_p\bigr)\,x_p \;-\; F
\]

---

**Constraints**

\[
\begin{aligned}
&\text{Labor capacity:}\qquad &&\sum_{p\in P} L_p\,x_p \;\le\; 1650\\[4pt]
&\text{Material capacity:}\qquad &&\sum_{p\in P} M_p\,x_p \;\le\; 1850\\[4pt]
&\text{Non‑negativity:}\qquad &&x_p \;\ge\; 0 \quad \forall p\in P
\end{aligned}
\]

---

**Data Mapping Rules**

| Symbol | Source Table | Source Column | Mapping Rule |
|--------|--------------|---------------|--------------|
| \(P\) | `file_0_view_0` | `Product Name` | \(P = \{\,\text{Product Name}\,\}\) |
| \(L_p\) | `file_0_view_0` | `Labor per unit` | \(L_p =\) value of `Labor per unit` for product \(p\) |
| \(M_p\) | `file_0_view_0` | `Material per unit` | \(M_p =\) value of `Material per unit` for product \(p\) |
| \(S_p\) | `file_0_view_0` | `Selling Price` | \(S_p =\) value of `Selling Price` for product \(p\) |
| \(V_p\) | `file_0_view_0` | `Variable Cost` | \(V_p =\) value of `Variable Cost` for product \(p\) |

All other constants (labor capacity 1650, material capacity 1850, fixed cost 4500) are hard‑coded in the model. No additional capacity limits or bounds are imposed beyond those specified.