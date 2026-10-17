**Mathematical Model**

Let  

- \(P=\{F1,\dots ,F15\}\) : set of candidate plants  
- \(C=\{C1,\dots ,C15\}\) : set of customers  

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
&y_{p}\in\{0,1\}\quad &&p\in P &&\text{(plant opened)}\\
&x_{p,c}\ge 0\quad &&p\in P,\;c\in C &&\text{(units shipped)}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{p\in P}\text{fixed\_cost}_{p}\,y_{p}
      +\sum_{p\in P}\sum_{c\in C}\text{transport\_cost}_{p,c}\,x_{p,c}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{p\in P}x_{p,c}= \text{demand}_{c}\qquad \forall\,c\in C
   \]

2. **Plant capacity**  
   \[
   \sum_{c\in C}x_{p,c}\;\le\; \text{capacity}_{p}\,y_{p}\qquad \forall\,p\in P
   \]

3. **Non‑negativity**  
   \[
   x_{p,c}\ge 0\qquad \forall\,p\in P,\;c\in C
   \]

4. **Binary opening decision**  
   \[
   y_{p}\in\{0,1\}\qquad \forall\,p\in P
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Mapping |
|-----------|--------------|---------------|---------------|
| `fixed_cost_p` | `file_0_view_0` | `fixed_cost` | `plant` → `p` |
| `capacity_p` | `file_0_view_0` | `capacity` | `plant` → `p` |
| `transport_cost_p_c` | `file_0_view_0` | `C1`–`C15` | `plant` → `p`, `C1`–`C15` → `c` |
| `demand_c` | `file_1_view_0` | `demand` | `customer` → `c` |

All indices (`p` and `c`) are taken directly from the `plant` and `customer` columns of the respective tables. No numerical values are included in the model; they are supplied by the data mapping above.