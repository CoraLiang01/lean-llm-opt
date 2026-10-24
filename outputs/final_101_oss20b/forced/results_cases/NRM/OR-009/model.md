**Mathematical Model**

Let  

- \(P=\{p\}\) be the set of production plants (rows of the cost matrix).  
- \(C=\{c\}\) be the set of retail customers (columns of the cost matrix).  

**Parameters**  

\[
\begin{aligned}
d_c &\quad\text{daily demand of customer }c,\\
s_p &\quad\text{production capacity of plant }p,\\
c_{p,c} &\quad\text{transportation cost per unit from plant }p\text{ to customer }c.
\end{aligned}
\]

**Decision Variables**  

\[
x_{p,c}\;\;(\ge 0,\;\text{continuous})\quad\text{quantity shipped from plant }p\text{ to customer }c.
\]

**Objective**  

\[
\min \;\sum_{p\in P}\sum_{c\in C} c_{p,c}\,x_{p,c}.
\]

**Constraints**

1. **Supply capacity**  
   \[
   \sum_{c\in C} x_{p,c}\;\le\; s_p,\qquad \forall p\in P.
   \]

2. **Demand satisfaction**  
   \[
   \sum_{p\in P} x_{p,c}\;=\; d_c,\qquad \forall c\in C.
   \]

3. **Non‑negativity**  
   \[
   x_{p,c}\;\ge\;0,\qquad \forall p\in P,\;c\in C.
   \]

---

### Data Mapping

| Parameter | Source Table | Source Column | ID Column |
|-----------|--------------|---------------|-----------|
| \(d_c\) | `file_0_view_0` | `demand` | `customer` |
| \(s_p\) | `file_1_view_0` | `supply_capacity` | `Unnamed: 0` |
| \(c_{p,c}\) | `file_2_view_0` | `C1`, `C2`, `C3`, `C4` | `Unnamed: 0` (rows) |

*All indices \(p\) and \(c\) are taken from the row and column identifiers of the respective tables.*