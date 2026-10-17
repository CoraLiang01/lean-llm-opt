**Mathematical Model**

Let  

* \(I\) – set of FAUX products (identified by *Product Name*).  
* For each \(i\in I\):  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*Objective*  

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

*Constraints*  

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I \quad &&\text{(cannot exceed demand)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I \quad &&\text{(cannot exceed inventory)}\\[4pt]
x_i &\;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(I\) (index) | `file_0_view_0` | `Product Name` | Each distinct FAUX product name |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit |
| \(d_i\) | `file_0_view_0` | `Demand` | Demand quantity |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Available inventory |

All values are taken directly from the filtered rows of `ZARASales.csv` where *Product Name* contains `"FAUX"`. No additional data or aggregation is introduced.