##### Abstract Model

**Index Set**  
\(I = \{i \mid \text{product } i \text{ in the dataset}\}\)

**Parameters**  
\(r_i \;(\text{revenue per unit of product } i)\)  
\(d_i \;(\text{deterministic demand for product } i)\)  
\(c_i \;(\text{initial inventory of product } i)\)

**Decision Variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) for all \(i \in I\)

**Objective Function**  
\[
\quad \max \quad \sum_{i \in I} r_i \, x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
\quad x_i \le c_i , \qquad \forall i \in I
\]

2. **Demand Constraints**  
\[
\quad x_i \le d_i , \qquad \forall i \in I
\]

3. **Variable Domain**  
\[
\quad x_i \in \mathbb{Z}_{\ge 0} , \qquad \forall i \in I
\]

---

##### Data Mapping

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(r_i\)   | file_0_view_0   | Revenue       |
| \(d_i\)   | file_0_view_0   | Demand        |
| \(c_i\)   | file_0_view_0   | Initial Inventory |

---