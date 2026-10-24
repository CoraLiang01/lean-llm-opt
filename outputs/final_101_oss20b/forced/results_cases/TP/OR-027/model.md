**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of *Organ* products (identified by the value in the **Sub Category** column).  
- \(r_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
- \(d_i\) – demand for product \(i\) (from column **Demand**).  
- \(s_i\) – initial inventory of product \(i\) (from column **Initial Inventory**).  
- \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column Name          | Mapping |
|-----------|--------------|----------------------|---------|
| \(I\)     | file_0_view_0 | **Sub Category**     | Each distinct value in this column defines an index \(i\). |
| \(r_i\)   | file_0_view_0 | **Revenue**          | \(r_i =\) value in the *Revenue* column for row \(i\). |
| \(d_i\)   | file_0_view_0 | **Demand**           | \(d_i =\) value in the *Demand* column for row \(i\). |
| \(s_i\)   | file_0_view_0 | **Initial Inventory**| \(s_i =\) value in the *Initial Inventory* column for row \(i\). |
| \(x_i\)   | –            | –                    | Decision variable, non‑negative continuous. |

All data are taken directly from the filtered view `file_0_view_0`; no additional aggregation or transformation is performed.