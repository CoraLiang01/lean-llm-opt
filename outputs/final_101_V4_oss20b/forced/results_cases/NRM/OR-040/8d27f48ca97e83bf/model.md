**Mathematical Model**

Let  

- \(I\) be the set of development areas (rows of *products.csv*).  
- For each \(i\in I\) let  

  \[
  b_i \;=\; \text{benefit coefficient of area } i \quad (\text{from column ``Value''})
  \]

- Let  

  \[
  C \;=\; \text{overall development capacity} \quad (\text{from column ``Capacity''})
  \]

- Decision variables  

  \[
  x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
  \]

**Objective**

\[
\max \;\sum_{i\in I} b_i\,x_i
\]

**Constraints**

1. **Capacity constraint**

   \[
   \sum_{i\in I} x_i \;\le\; C
   \]

2. **Non‑negativity and integrality**

   \[
   x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\qquad \forall i\in I
   \]

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(b_i\)   | `file_1_view_0` | `Value` |
| \(C\)     | `file_0_view_0` | `Capacity` |

*Index set \(I\) is derived from the distinct `ProductName` values in `file_1_view_0`.*