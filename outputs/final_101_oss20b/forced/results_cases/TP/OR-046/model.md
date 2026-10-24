**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of products (index \(i\) corresponds to the \(i\)-th row of *products.csv*).  
* \(x_i\ge 0\) – number of units of product \(i\) ordered each day.  

Parameters  

* \(w_i\) – weight of one unit of product \(i\) (kg).  
* \(v_i\) – value (benefit) of one unit of product \(i\).  
* \(C\) – total stock capacity (kg).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column(s) | Notes |
|-----------|--------|----------|-----------|-------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` | Single value |
| \(w_i\) | Weight | `file_1_view_0` | `Weight` | One row per product |
| \(v_i\) | Value | `file_1_view_0` | `Value` | One row per product |

*Index mapping:*  
Row \(k\) (0‑based) of `file_1_view_0` corresponds to product index \(i=k+1\).  
The `ProductName` column is only for reference and is not used in the model.