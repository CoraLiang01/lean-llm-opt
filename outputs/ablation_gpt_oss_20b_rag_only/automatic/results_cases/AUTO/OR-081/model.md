**Optimization Model**

\[
\begin{aligned}
\min_{x}\;& Z = \sum_{f\in F} \text{Cost}_f\,x_f \\[4pt]
\text{s.t. }&
\sum_{f\in F} \text{Calories}_f\,x_f \;\ge\; 2000 \\[2pt]
&\sum_{f\in F} \text{Protein}_f\,x_f \;\ge\; 50 \\[2pt]
&\sum_{f\in F} \text{VitaminC}_f\,x_f \;\ge\; 60 \\[2pt]
&\sum_{f\in F} \text{Fat}_f\,x_f \;\le\; 70 \\[4pt]
&x_f \;\ge\; 0 \qquad \forall f\in F
\end{aligned}
\]

* \(F\) – set of all food items (120 items).  
* \(x_f\) – number of servings of food \(f\) (continuous, non‑negative).  
* Parameters \(\text{Calories}_f,\text{Protein}_f,\text{Fat}_f,\text{VitaminC}_f,\text{Cost}_f\) are taken from the data table.

---

**Data Mapping**

| Parameter | Source Table | Column Name | Notes |
|-----------|--------------|-------------|-------|
| \(\text{Calories}_f\) | `file_0_view_0` | `Calories` | Float |
| \(\text{Protein}_f\) | `file_0_view_0` | `Protein(g)` | Float |
| \(\text{Fat}_f\) | `file_0_view_0` | `Fat(g)` | Float |
| \(\text{VitaminC}_f\) | `file_0_view_0` | `VitaminC(mg)` | Float |
| \(\text{Cost}_f\) | `file_0_view_0` | `Cost` | Float |
| Index set \(F\) | `file_0_view_0` | `Food` | Each distinct value defines an index \(f\). |

*No filters are applied; the decision set includes all 120 rows returned by the observation.*