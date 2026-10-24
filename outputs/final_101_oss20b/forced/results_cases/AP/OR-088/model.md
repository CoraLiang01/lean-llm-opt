**Mathematical Model**

Let  

- \(P=\{F1,\dots ,F15\}\) (plant indices)  
- \(C=\{C1,\dots ,C15\}\) (customer indices)

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_{p}\quad &&p\in P\\
&\text{capacity}_{p}\quad &&p\in P\\
&\text{transport\_cost}_{p,c}\quad &&p\in P,\;c\in C\\
&\text{demand}_{c}\quad &&c\in C
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&x_{p}\in\{0,1\}\quad &&p\in P &&\text{(plant \(p\) is opened)}\\
&y_{p,c}\ge 0\quad &&p\in P,\;c\in C &&\text{(units shipped from \(p\) to \(c\))}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{p\in P}\text{fixed\_cost}_{p}\,x_{p}
      +\sum_{p\in P}\sum_{c\in C}\text{transport\_cost}_{p,c}\,y_{p,c}
\]

**Constraints**

1. **Capacity of opened plants**

\[
\sum_{c\in C} y_{p,c}\;\le\;\text{capacity}_{p}\,x_{p}\qquad \forall p\in P
\]

2. **Demand satisfaction**

\[
\sum_{p\in P} y_{p,c}\;=\;\text{demand}_{c}\qquad \forall c\in C
\]

3. **Variable domains**

\[
x_{p}\in\{0,1\}\;\;\forall p\in P,\qquad
y_{p,c}\ge 0\;\;\forall p\in P,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| `fixed_cost_p` | `file_0_view_0` | `fixed_cost` |
| `capacity_p` | `file_0_view_0` | `capacity` |
| `transport_cost_p_c` | `file_0_view_0` | `C1` … `C15` |
| `demand_c` | `file_1_view_0` | `demand` |

All indices \(p\) and \(c\) are taken from the corresponding rows of the two CSV files. No numerical values are listed; the model references the data via the table and column names above.