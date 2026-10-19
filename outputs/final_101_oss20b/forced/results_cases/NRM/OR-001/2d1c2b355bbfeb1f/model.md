**Abstract Model**

**Index Sets**  
\(I\) – set of customer groups (customers)  
\(J\) – set of distribution centers (supply centers)

**Parameters**  
\(d_i\) – daily demand of customer \(i\)  \(i \in I\)  
\(s_j\) – daily supply capacity of distribution center \(j\)  \(j \in J\)  
\(c_{j,i}\) – transportation cost per unit from distribution center \(j\) to customer \(i\)  \(j \in J,\; i \in I\)

**Decision Variables**  
\(x_{j,i}\) – quantity of goods transported from distribution center \(j\) to customer \(i\)  \(j \in J,\; i \in I\)  
\(x_{j,i} \in \mathbb{Z}_{+}\)

**Objective Function**  
\[
\min \;\sum_{j \in J}\;\sum_{i \in I}\; c_{j,i}\;x_{j,i}
\]

**Constraints**

1. **Demand Satisfaction**  
\[
\sum_{j \in J} x_{j,i} = d_i , \qquad \forall i \in I
\]

2. **Supply Capacity**  
\[
\sum_{i \in I} x_{j,i} \le s_j , \qquad \forall j \in J
\]

3. **Non‑negativity and Integrality**  
\[
x_{j,i} \in \mathbb{Z}_{+}, \qquad \forall j \in J,\; i \in I
\]

---

### Data Mapping

| Parameter | Source Table ID | Source Column Name |
|-----------|-----------------|--------------------|
| \(d_i\) | `file_0_view_0` | `demand` |
| \(s_j\) | `file_1_view_0` | `supply_capacity` |
| \(c_{j,i}\) | `file_2_view_0` | `C1` … `C18` (columns corresponding to customers) |

*The table IDs correspond to the CSV files:*

- `file_0_view_0` – *customer_demand.csv*  
- `file_1_view_0` – *supply_capacity.csv*  
- `file_2_view_0` – *transportation_costs.csv*  

*No numerical values are included in this abstract formulation; the model is ready for parameter instantiation from the specified tables.*