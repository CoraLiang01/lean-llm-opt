**Symbolic Model**

Let  

- \(I=\{S_1,\dots ,S_{24}\}\) – set of suppliers  
- \(J=\{C_1,\dots ,C_{25}\}\) – set of supermarkets  

Parameters  

\[
\begin{aligned}
\text{fixed\_cost}_i &\quad (i\in I)\\
\text{demand}_j &\quad (j\in J)\\
\text{transport\_cost}_{ij} &\quad (i\in I,\;j\in J)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(supplier }i\text{ is opened)}\\
x_{ij} &\ge 0 &&\text{(units supplied from }i\text{ to }j)
\end{aligned}
\]

---

**Objective**

\[
\min \;\; \sum_{i\in I}\text{fixed\_cost}_i\,y_i
      \;+\; \sum_{i\in I}\sum_{j\in J}\text{transport\_cost}_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; \text{demand}_j \qquad \forall j\in J
   \]

2. **Supplier‑activation linkage**  
   \[
   x_{ij} \;\le\; \text{demand}_j \; y_i \qquad \forall i\in I,\; j\in J
   \]

   (If supplier \(i\) is closed (\(y_i=0\)), no flow can leave it.)

---

**Data Mapping**

| Parameter | Source Table | Column | Row Identifier |
|-----------|--------------|--------|----------------|
| `fixed_cost_i` | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` (values `S1`…`S24`) |
| `demand_j` | `file_0_view_0` | `demand` | `customer` (values `C1`…`C25`) |
| `transport_cost_ij` | `file_2_view_0` | `Cj` (column matching supermarket `j`) | `Unnamed: 0` (values `S1`…`S24`) |

All numerical values are taken directly from the corresponding CSV files; no additional data or assumptions are introduced.