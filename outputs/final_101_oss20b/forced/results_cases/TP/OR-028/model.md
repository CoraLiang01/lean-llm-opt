**Mathematical Model**

Let  

- \(P\) be the set of products (indexed by the *Product Name* values).  
- For each \(p\in P\)  
  - \(rev_p\)  – revenue per unit (from column **Revenue**).  
  - \(dem_p\)  – demand quantity (from column **Demand**).  
  - \(inv_p\)  – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_p \;\ge\; 0 \qquad \forall p\in P
\]

(Units of product \(p\) to be sold.)

---

**Objective**

\[
\max \;\sum_{p\in P} rev_p \, x_p
\]

---

**Constraints**

\[
\begin{aligned}
&x_p \;\le\; dem_p &&\forall p\in P \quad &&\text{(cannot sell more than demand)}\\
&x_p \;\le\; inv_p &&\forall p\in P \quad &&\text{(cannot sell more than inventory)}\\
&x_p \;\ge\; 0   &&\forall p\in P
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Column Name |
|----------------------|--------------|-------------|
| \(P\) (index set)   | `file_0_view_0` | `Product Name` |
| \(rev_p\)            | `file_0_view_0` | `Revenue` |
| \(dem_p\)            | `file_0_view_0` | `Demand` |
| \(inv_p\)            | `file_0_view_0` | `Initial Inventory` |
| \(x_p\)              | – (decision variable) | – |

*All numeric values are taken directly from the corresponding columns of the table `WomenClothingEcommerceSalesData.csv` (view `file_0_view_0`).*