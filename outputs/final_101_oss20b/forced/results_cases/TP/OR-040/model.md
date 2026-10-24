**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of real‑estate areas (the `ProductName` values in *products.csv*).  
* \(b_i\) be the benefit coefficient of area \(i\) (the `Value` column).  
* \(C\) be the overall development capacity (the `Capacity` column in *capacity.csv*).  
* \(x_i\in\mathbb{Z}_{\ge 0}\) be the daily scale of development in area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Source | Table ID | Column(s) | Mapping |
|--------|----------|-----------|---------|
| *capacity.csv* | `file_0_view_0` | `Capacity` | `C = Capacity` |
| *products.csv* | `file_1_view_0` | `ProductName`, `Value` | For each record: <br> * index \(i =\) `ProductName` <br> * benefit coefficient \(b_i =\) `Value` |

*The `Weight` column in *products.csv* is not used in the model.*