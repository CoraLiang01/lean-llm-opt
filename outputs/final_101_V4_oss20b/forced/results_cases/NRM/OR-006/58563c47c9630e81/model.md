**Symbolic Model**

Let  

- \(W=\{S1,\dots ,S10\}\) (warehouses)  
- \(C=\{C1,\dots ,C10\}\) (customers)

Parameters  

\[
\begin{aligned}
d_c &\quad\text{(daily demand of customer }c\text{)}\\
s_w &\quad\text{(daily supply capacity of warehouse }w\text{)}\\
c_{w,c} &\quad\text{(unit transportation cost from }w\text{ to }c\text{)}
\end{aligned}
\]

Decision variables  

\[
x_{w,c}\;\ge 0 \qquad\forall w\in W,\;c\in C
\]

---

**Objective**

\[
\min \;\; \sum_{w\in W}\;\sum_{c\in C}\; c_{w,c}\,x_{w,c}
\]

---

**Constraints**

1. **Supply capacity**

\[
\sum_{c\in C} x_{w,c}\;\le\; s_w \qquad\forall w\in W
\]

2. **Demand satisfaction**

\[
\sum_{w\in W} x_{w,c}\;=\; d_c \qquad\forall c\in C
\]

3. **Non‑negativity**

\[
x_{w,c}\;\ge\;0 \qquad\forall w\in W,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(d_c\)   | `file_0_view_0` | `demand` |
| \(s_w\)   | `file_1_view_0` | `supply_capacity` |
| \(c_{w,c}\) | `file_2_view_0` | matrix entry at row `Unnamed: 0 = w`, column `C? = c` |

All indices \(w\) and \(c\) are taken from the row and column identifiers of the respective tables. No additional data or assumptions are introduced beyond what is supplied in the observation.