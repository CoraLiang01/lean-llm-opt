**Mathematical Model**

Let  

* \(I=\{1,\dots ,20\}\) be the set of drug products (index by the row order in *products.csv*).  
* \(x_i\in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) ordered each day.  

Parameters  

* \(v_i\) – benefit per unit of product \(i\) (column **Value** in *products.csv*).  
* \(w_i\) – weight per unit of product \(i\) (column **Weight** in *products.csv*).  
* \(C\) – total stock‑capacity limit (column **Capacity** in *capacity.csv*).  

The optimization problem is  

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(C\) | `file_0_view_0` | `Capacity` | Single scalar value |
| \(v_i\) | `file_1_view_0` | `Value` | One value per product row |
| \(w_i\) | `file_1_view_0` | `Weight` | One value per product row |
| Index set \(I\) | `file_1_view_0` | row index | All 20 rows returned |

All parameters are taken directly from the corresponding columns of the returned rows; no additional filtering or aggregation is applied. The decision variables \(x_i\) are defined for every product in \(I\) and are constrained to be non‑negative integers. The single capacity constraint uses the weight coefficients \(w_i\) and the capacity limit \(C\). The objective is to maximize the total benefit \(\sum v_i x_i\).