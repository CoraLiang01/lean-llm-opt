#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products classified under ‘ZZ’, indexed by $i$ (see Data Mapping for SKU).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from Revenue column).
- $d_i$: Demand for product $i$ (from Demand column).
- $s_i$: Initial Inventory of product $i$ (from Initial Inventory column).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** $x_i \leq d_i \quad \forall i \in I$
2. **Inventory limit:**   $x_i \leq s_i \quad \forall i \in I$
3. **Nonnegativity and integrality:** $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

#### Data Mapping

| Index $i$ | SKU (file_0_view_0.SKU) | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
|-----------|-------------------------|-----------------|----------------|--------------------------|
| 1         | ZZ2AO                   | 24.38           | 2              | 10.0                     |
| 2         | ZZDW7                   | 30.12           | 4              | 20.0                     |
| 3         | ZZM1A                   | 19.52           | 82             | 530.0                    |
| 4         | ZZNC5                   | 10.79           | 2              | 10.0                     |
| 5         | ZZX6K                   | 111.81          | 2              | 10.0                     |

- All parameters are mapped directly from file_0_view_0 (RetailStoreSalesTransactions(ScannerData).csv) using the SKU as the business identifier.
- $x_i$ is the number of units of SKU $i$ to fulfill, subject to $0 \leq x_i \leq \min\{d_i, s_i\}$ and integer.

---

**Summary:**  
Maximize total revenue from fulfilling sales of ‘ZZ’ products, subject to initial inventory and demand constraints, with all variables and parameters mapped directly from the supplied data.