**Mathematical Model**

Let  

* \(I=\{i\mid \text{Sub Category of }i\text{ contains the string “Organ”}\}\)  
  (here \(I=\{\text{Organic Fruits},\text{Organic Staples},\text{Organic Vegetables}\}\))  

Parameters (all taken from the filtered view `file_0_view_0`):

* \(\text{Revenue}_i\) – revenue per unit of product \(i\)  
* \(\text{Demand}_i\) – deterministic demand for product \(i\)  
* \(\text{Inv}_i\) – initial inventory of product \(i\)

Decision variables:

* \(x_i\ge 0\) – number of units of product \(i\) that will be sold (fulfilled)

---

\[
\begin{aligned}
\max\;& \sum_{i\in I}\text{Revenue}_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; \text{Demand}_i &&\forall i\in I \\[2pt]
& x_i \;\le\; \text{Inv}_i &&\forall i\in I \\[2pt]
& x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` | For each row \(i\) in the filtered view, assign the value of `Revenue` to \(\text{Revenue}_i\). |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` | For each row \(i\) in the filtered view, assign the value of `Demand` to \(\text{Demand}_i\). |
| \(\text{Inv}_i\) | `file_0_view_0` | `Initial Inventory` | For each row \(i\) in the filtered view, assign the value of `Initial Inventory` to \(\text{Inv}_i\). |
| Index set \(I\) | `file_0_view_0` | `Sub Category` | Include a row in \(I\) if its `Sub Category` value contains the substring “Organ”. The row’s identifier (e.g., the full `Sub Category` string) is used as the index \(i\). |

All parameters are taken directly from the filtered view; no additional numerical values are introduced. The model is fully specified by the above symbolic formulation and data‑mapping rules.