**Mathematical Model**

Let  

* \(I\) be the set of products whose *Product Name* starts with the prefix **“Aalop”**.  
* For each \(i\in I\)  
  * \(R_i\) – revenue per unit (from column **Revenue**).  
  * \(D_i\) – known demand during the sales horizon (from column **Demand**).  
  * \(I_i\) – initial inventory available at the start of the horizon (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that will be sold.

**Objective**

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; D_i &&\forall i\in I \quad &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; I_i &&\forall i\in I \quad &&\text{(cannot sell more than initial inventory)}\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

### Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(R_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand during horizon | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | Product names with prefix “Aalop” | `file_0_view_0` | `Product Name` |

No additional parameters or constraints are required. The model uses only the data returned by the query‑filtered view.