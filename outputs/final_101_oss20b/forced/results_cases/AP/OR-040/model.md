**Mathematical Model**

Let  

* \(I\) – set of development areas (indices are the values of *ProductName*).  
* \(b_i\) – benefit coefficient for area \(i\) (taken from the *Value* column).  
* \(C\) – overall development capacity (taken from the *Capacity* column).  
* \(x_i\) – integer number of development units scheduled in area \(i\) each day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Set | Source | Table ID | Column(s) | Mapping Rule |
|-----------------|--------|----------|-----------|--------------|
| \(I\) | Products | `file_1_view_0` | `ProductName` | \(I = \{\,\text{ProductName}\,\}\) |
| \(b_i\) | Products | `file_1_view_0` | `Value` | \(b_i = \text{Value}\) for the row where `ProductName = i\) |
| \(C\) | Capacity | `file_0_view_0` | `Capacity` | \(C = \text{Capacity}\) (single row) |

*All other columns (e.g., `Weight`) are not used in this model.*