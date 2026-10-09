**Abstract Mathematical Model**

**Index Sets:**
- $I$ : Set of all products in table_id = "file_0_view_0" such that the value in column "Product Name" contains the substring "Baby".

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$, from column "Revenue" in table_id = "file_0_view_0".
- $d_i$ : Demand for product $i \in I$, from column "Demand" in table_id = "file_0_view_0".
- $I_i$ : Initial inventory for product $i \in I$, from column "Initial Inventory" in table_id = "file_0_view_0".

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory and Demand Bounds:**
   \[
   0 \leq x_i \leq \min\{d_i, I_i\} \quad \forall i \in I
   \]
   (Implemented as two separate constraints:)
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping:**

- **Source Table:** table_id = "file_0_view_0" (file_name = "EuropeSalesRecords.csv")
- **Product Selection:** $I$ is the set of all records where "Product Name" contains the substring "Baby".
- **Revenue Parameter:** $A_i$ is the value in column "Revenue" for product $i$.
- **Demand Parameter:** $d_i$ is the value in column "Demand" for product $i$.
- **Initial Inventory Parameter:** $I_i$ is the value in column "Initial Inventory" for product $i$.
- **No additional filters or eligibility conditions are applied beyond the substring match on "Product Name".

---

**Summary:**  
This model maximizes total revenue from all products classified as ‘Baby’ in the provided table, subject to deterministic demand and initial inventory constraints, using integer fulfillment decisions. All data is mapped directly from the specified columns and table, with explicit selection of ‘Baby’ products.