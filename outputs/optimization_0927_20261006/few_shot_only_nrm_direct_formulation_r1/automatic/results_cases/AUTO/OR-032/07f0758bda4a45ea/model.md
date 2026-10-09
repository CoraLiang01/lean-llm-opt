**Index Set:**  
Let $\mathcal{B}$ be the set of all "Books" products:  
$\mathcal{B} = \{$  
$\quad$ "Books\_15.15",  
$\quad$ "Books\_30.3",  
$\quad$ "Books\_45.45",  
$\quad$ "Books\_60.6",  
$\quad$ "Books\_75.75"  
$\}$

**Parameters:**  
For each $i \in \mathcal{B}$:
- $A_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $I_i$: Initial Inventory for product $i$

| $i$                   | $A_i$   | $d_i$ | $I_i$    |
|-----------------------|---------|-------|----------|
| Books\_15.15          | 15.15   | 1980  | 9920.0   |
| Books\_30.3           | 30.3    | 3024  | 20160.0  |
| Books\_45.45          | 45.45   | 4536  | 30000.0  |
| Books\_60.6           | 60.6    | 5601  | 38360.0  |
| Books\_75.75          | 75.75   | 7567  | 51450.0  |

**Decision Variables:**  
For each $i \in \mathcal{B}$:  
$x_i =$ number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{B}} A_i \cdot x_i
$$

**Constraints:**  
For each $i \in \mathcal{B}$:
1. Inventory constraint:  
   $x_i \leq I_i$
2. Demand constraint:  
   $x_i \leq d_i$
3. Non-negativity and integrality:  
   $x_i \in \mathbb{Z}_+, \quad x_i \geq 0$

**Complete Model (Numerical Formulation):**

**Parameters:**
- $\mathcal{B} = \{$"Books\_15.15", "Books\_30.3", "Books\_45.45", "Books\_60.6", "Books\_75.75"$\}$
- $A = [15.15, 30.3, 45.45, 60.6, 75.75]$ (in source order)
- $d = [1980, 3024, 4536, 5601, 7567]$
- $I = [9920.0, 20160.0, 30000.0, 38360.0, 51450.0]$

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{B}$

**Objective:**
$$
\max \left(
15.15\, x_{\text{Books\_15.15}} +
30.3\, x_{\text{Books\_30.3}} +
45.45\, x_{\text{Books\_45.45}} +
60.6\, x_{\text{Books\_60.6}} +
75.75\, x_{\text{Books\_75.75}}
\right)
$$

**Subject to:**
\[
\begin{align*}
x_{\text{Books\_15.15}} &\leq 1980 \\
x_{\text{Books\_15.15}} &\leq 9920.0 \\
x_{\text{Books\_30.3}} &\leq 3024 \\
x_{\text{Books\_30.3}} &\leq 20160.0 \\
x_{\text{Books\_45.45}} &\leq 4536 \\
x_{\text{Books\_45.45}} &\leq 30000.0 \\
x_{\text{Books\_60.6}} &\leq 5601 \\
x_{\text{Books\_60.6}} &\leq 38360.0 \\
x_{\text{Books\_75.75}} &\leq 7567 \\
x_{\text{Books\_75.75}} &\leq 51450.0 \\
x_i &\in \mathbb{Z}_+, \quad \forall i \in \mathcal{B}
\end{align*}
\]

**Retrieved Information (for code generation):**

- Index set:  
  $\mathcal{B} =$ ["Books\_15.15", "Books\_30.3", "Books\_45.45", "Books\_60.6", "Books\_75.75"]

- Revenue vector $A$ (per unit, in source order):  
  [15.15, 30.3, 45.45, 60.6, 75.75]

- Demand vector $d$ (in source order):  
  [1980, 3024, 4536, 5601, 7567]

- Initial Inventory vector $I$ (in source order):  
  [9920.0, 20160.0, 30000.0, 38360.0, 51450.0]

- Variable domains:  
  $x_i \in \mathbb{Z}_+, \forall i \in \mathcal{B}$

**End of Model.**