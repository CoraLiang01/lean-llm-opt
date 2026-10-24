**Mathematical Model**

Let  

- \(I=\{F1,\dots ,F15\}\) (plant indices)  
- \(J=\{C1,\dots ,C15\}\) (customer indices)

**Parameters**

\[
\begin{aligned}
&\text{fixed\_cost}_i &&\forall i\in I\\
&\text{capacity}_i   &&\forall i\in I\\
&\text{transport}_{ij}&&\forall i\in I,\;j\in J\\
&\text{demand}_j     &&\forall j\in J
\end{aligned}
\]

**Decision Variables**

\[
\begin{aligned}
&y_i \in \{0,1\} &&\forall i\in I &&\text{(plant opened)}\\
&x_{ij}\ge 0 &&\forall i\in I,\;j\in J &&\text{(units shipped)}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
      +\sum_{i\in I}\sum_{j\in J}\text{transport}_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I}x_{ij}= \text{demand}_j,\qquad \forall j\in J
   \]

2. **Capacity of opened plants**  
   \[
   \sum_{j\in J}x_{ij}\le \text{capacity}_i\,y_i,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\quad \forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| `fixed_cost_i` | `file_0_view_0` | `fixed_cost` | `fixed_cost_i = value where plant = i` |
| `capacity_i`   | `file_0_view_0` | `capacity`   | `capacity_i = value where plant = i` |
| `transport_{ij}` | `file_0_view_0` | `C1`–`C15` | `transport_{ij} = value where plant = i and column = Cj` |
| `demand_j` | `file_1_view_0` | `demand` | `demand_j = value where customer = j` |

All index sets \(I\) and \(J\) are taken from the distinct values of the `plant` and `customer` columns in the respective tables. No additional numerical limits are imposed beyond those supplied.