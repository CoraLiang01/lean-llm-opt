**Symbolic Model**

Let  

* \(S=\{S1,S2,\dots ,S10\}\) (warehouses)  
* \(C=\{C1,C2,\dots ,C10\}\) (customers)

Parameters  

\[
\begin{aligned}
d_c &\quad\text{(daily demand of customer }c)\\
s_s &\quad\text{(daily supply capacity of warehouse }s)\\
c_{s,c} &\quad\text{(unit transportation cost from }s\text{ to }c)
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{+}\qquad\forall s\in S,\;c\in C
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{s\in S} x_{s,c}=d_c \qquad\forall c\in C
\]

2. **Supply capacity**  
\[
\sum_{c\in C} x_{s,c}\le s_s \qquad\forall s\in S
\]

3. **Non‑negativity & integrality**  
\[
x_{s,c}\in \mathbb{Z}_{+}\qquad\forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(d_c\) | `file_0_view_0` | `demand` |
| \(s_s\) | `file_1_view_0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `C1`, `C2`, `C3`, `C4` (for corresponding customer) |
| Warehouse ID | `file_1_view_0` and `file_2_view_0` | `Unnamed: 0` |
| Customer ID | `file_0_view_0` | `customer` |

*Only the entries present in the source tables are defined; any missing cost entries are left undefined in the model.*