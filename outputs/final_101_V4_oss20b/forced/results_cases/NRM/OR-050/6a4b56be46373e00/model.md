**Abstract Model**

**Index Sets**  
\(I\) – set of displays (indexed by \(i\)).  
\(J\) – set of products (indexed by \(j\)).  

**Parameters**  
\(C_i\) – capacity of display \(i\) (units of weight).  
\(V_j\) – value of product \(j\) (currency units).  
\(W_j\) – weight of product \(j\) (weight units).  

**Decision Variables**  
\(x_{ij}\) – number of units of product \(j\) placed on display \(i\).  

**Objective Function**  
\[
\max \sum_{i\in I}\sum_{j\in J} V_j \, x_{ij}
\]

**Constraints**  

1. **Weight Capacity Constraints**  
\[
\sum_{j\in J} W_j \, x_{ij} \;\le\; C_i , \qquad \forall i\in I
\]

2. **First‑Product Minimum Quantity**  
Let \(j^*\) denote the index of the first product (“Smartphone”).  
\[
\sum_{i\in I} x_{i j^*} \;\ge\; 5
\]

3. **Non‑negativity and Integrality**  
\[
x_{ij} \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(C_i\) | `file_0_view_0` (capacity.csv) | `Capacity` |
| \(V_j\) | `file_1_view_0` (products.csv) | `Value` |
| \(W_j\) | `file_1_view_0` (products.csv) | `Weight` |

*The first product (“Smartphone”) is identified by the record in `file_1_view_1` where `ProductName = 'Smartphone'`. Its index \(j^*\) is used in the minimum‑quantity constraint.*