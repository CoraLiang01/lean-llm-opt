**Mathematical Model**

**Index Sets:**
- $S$: Set of displays (shelves), indexed by $s$, with business key $\texttt{ShelfID}$ from $\texttt{file\_0\_view\_0}$.
- $P$: Set of products, indexed by $p$, with business key $\texttt{ProductName}$ from $\texttt{file\_1\_view\_0}$.

**Parameters:**
- $C_s$: Capacity of display $s$ (from $\texttt{file\_0\_view\_0}$, column $\texttt{Capacity}$).
- $v_p$: Value of product $p$ (from $\texttt{file\_1\_view\_0}$, column $\texttt{Value}$).
- $w_p$: Weight of product $p$ (from $\texttt{file\_1\_view\_0}$, column $\texttt{Weight}$).

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on display $s$.

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**

1. **Display Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **Minimum Placement of First Product:**
   Let $p^*$ be the product with $\texttt{source\_row} = 0$ in $\texttt{file\_1\_view\_0}$ (i.e., the first product in the file).
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All $\texttt{ShelfID}$ in $\texttt{file\_0\_view\_0}$.
- $P$: All $\texttt{ProductName}$ in $\texttt{file\_1\_view\_0}$.
- $C_s$: $\texttt{Capacity}$ from $\texttt{file\_0\_view\_0}$, keyed by $\texttt{ShelfID}$.
- $v_p$: $\texttt{Value}$ from $\texttt{file\_1\_view\_0}$, keyed by $\texttt{ProductName}$.
- $w_p$: $\texttt{Weight}$ from $\texttt{file\_1\_view\_0}$, keyed by $\texttt{ProductName}$.
- $p^*$: The $\texttt{ProductName}$ with $\texttt{source\_row} = 0$ in $\texttt{file\_1\_view\_0}$.

---

**Summary:**  
Maximize total value of products allocated to displays, subject to each display's capacity, with at least 5 units of the first product (by file order) placed in total. All variables are nonnegative integers. All parameters and index sets are mapped directly to the supplied data columns and business keys.