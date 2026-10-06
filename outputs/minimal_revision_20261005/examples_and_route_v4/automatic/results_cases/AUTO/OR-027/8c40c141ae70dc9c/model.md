**Abstract Mathematical Model**

**Index Set:**
- $i \in I$ : Set of products with $\texttt{Sub Category}$ starting with "Organ" (from SupermartGrocerySales-RetailAnalyticsDataset.csv).

**Parameters:**
- $r_i$ : Revenue per unit of product $i$ ($\texttt{Revenue}$, from file_0_view_0, column "Revenue").
- $d_i$ : Demand for product $i$ ($\texttt{Demand}$, from file_0_view_0, column "Demand").
- $s_i$ : Initial inventory of product $i$ ($\texttt{Initial Inventory}$, from file_0_view_0, column "Initial Inventory").

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory constraint:**  
   $\quad x_i \leq s_i \quad \forall i \in I$

2. **Demand constraint:**  
   $\quad x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

**Data Mapping**

- $I$: All rows in SupermartGrocerySales-RetailAnalyticsDataset.csv where $\texttt{Sub Category}$ starts with "Organ" (see file_0_view_0).
- $r_i$: file_0_view_0, column "Revenue", keyed by $\texttt{Sub Category}$.
- $d_i$: file_0_view_0, column "Demand", keyed by $\texttt{Sub Category}$.
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by $\texttt{Sub Category}$.

**Variable Domain:**  
$x_i$ is a nonnegative integer for each $i \in I$.