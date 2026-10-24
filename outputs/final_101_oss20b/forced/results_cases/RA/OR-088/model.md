**Mathematical Model**

Let  

- \(P=\{F1,\dots ,F15\}\) (plant indices)  
- \(C=\{C1,\dots ,C15\}\) (customer indices)

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_{p}\quad &&p\in P\\
&\text{capacity}_{p}\quad &&p\in P\\
&\text{transport}_{p,c}\quad &&p\in P,\;c\in C\\
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
      +\sum_{p\in P}\sum_{c\in C}\text{transport}_{p,c}\,x_{p,c}
\]

**Constraints**

1. **Capacity of opened plants**

\[
\sum_{c\in C}x_{p,c}\;\le\;\text{capacity}_{p}\,y_{p}\qquad \forall p\in P
\]

2. **Demand satisfaction**

\[
\sum_{p\in P}x_{p,c}\;\ge\;\text{demand}_{c}\qquad \forall c\in C
\]

3. **Non‑negativity and integrality**

\[
x_{p,c}\ge 0,\; y_{p}\in\{0,1\}\qquad \forall p\in P,\;c\in C
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| fixed_cost | file_0_view_0 | fixed_cost |
| capacity | file_0_view_0 | capacity |
| transport | file_0_view_0 | C1–C15 (each column corresponds to a customer) |
| demand | file_1_view_0 | demand |

*Indices*  
- Plant index \(p\) is taken from the `plant` column of `file_0_view_0`.  
- Customer index \(c\) is taken from the `customer` column of `file_1_view_0`.  

All parameter values are read directly from the corresponding CSV rows; no values are enumerated in the model.