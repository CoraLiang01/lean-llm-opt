**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* starts with “Aalop”.  
- For each \(i\in I\)  
  - \(R_i\) = revenue per unit (from column **Revenue**)  
  - \(D_i\) = demand (from column **Demand**)  
  - \(S_i\) = initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\ge 0\) = units of product \(i\) to sell during the horizon.

\[
\begin{aligned}
\max\;&\sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } &x_i \;\le\; D_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\
            &x_i \;\le\; S_i &&\forall i\in I \quad\text{(cannot exceed initial inventory)}\\
            &x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(I\) | `file_0_view_0` | *Product Name* (filter: prefix “Aalop”) |
| \(R_i\) | `file_0_view_0` | **Revenue** |
| \(D_i\) | `file_0_view_0` | **Demand** |
| \(S_i\) | `file_0_view_0` | **Initial Inventory** |

The index set \(I\) is constructed by selecting all rows in `file_0_view_0` whose *Product Name* begins with “Aalop”. Each row supplies the corresponding values for \(R_i\), \(D_i\), and \(S_i\). No other tables or parameters are required.