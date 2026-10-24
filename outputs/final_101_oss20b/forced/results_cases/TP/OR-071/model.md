**Mathematical Model**

Let  

* \(P\) – set of all products (index \(p\)).  
* \(x_p \ge 0\) – continuous production quantity of product \(p\).

Parameters (obtained from *41.csv* and the problem statement)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(L_p\) | Labor required per unit of product \(p\) | column **Labor per unit** |
| \(M_p\) | Material required per unit of product \(p\) | column **Material per unit** |
| \(SP_p\) | Selling price per unit of product \(p\) | column **Selling Price** |
| \(VC_p\) | Variable cost per unit of product \(p\) | column **Variable Cost** |
| \(LC\) | Weekly labor capacity | 1650 |
| \(MC\) | Weekly material capacity | 1850 |
| \(FC\) | Fixed operating cost per week | 4500 |

---

\[
\begin{aligned}
\max_{x}\quad & \sum_{p\in P}\bigl(SP_p-VC_p\bigr)\,x_p \;-\; FC \\[4pt]
\text{s.t.}\quad
& \sum_{p\in P} L_p\,x_p \;\le\; LC \\[4pt]
& \sum_{p\in P} M_p\,x_p \;\le\; MC \\[4pt]
& x_p \;\ge\; 0 \qquad \forall p\in P
\end{aligned}
\]

---

**Data Mapping Rules**

| Index Set | Source | Notes |
|-----------|--------|-------|
| \(P\) | All distinct values of **Product Name** in *41.csv* | Each product is a unique index. |

| Parameter | Source Column | Table ID | Notes |
|-----------|---------------|----------|-------|
| \(L_p\) | **Labor per unit** | `file_0_view_0` | Float |
| \(M_p\) | **Material per unit** | `file_0_view_0` | Float |
| \(SP_p\) | **Selling Price** | `file_0_view_0` | Float |
| \(VC_p\) | **Variable Cost** | `file_0_view_0` | Float |
| \(LC\) | Constant | – | 1650 |
| \(MC\) | Constant | – | 1850 |
| \(FC\) | Constant | – | 4500 |

These mappings bind each symbolic element of the model to the exact column in the source CSV, ensuring that downstream solvers can retrieve the full numerical data set.