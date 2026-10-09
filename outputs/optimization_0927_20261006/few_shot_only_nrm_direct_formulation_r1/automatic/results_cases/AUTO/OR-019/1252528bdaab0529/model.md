**Index Set:**  
Let $I = \{$  
$\quad$ "27in 4K Gaming Monitor",  
$\quad$ "27in FHD Monitor"  
$\}$

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit  
  $A_{\text{"27in 4K Gaming Monitor"}} = 389.99$  
  $A_{\text{"27in FHD Monitor"}} = 149.99$
- $d_i$: Demand  
  $d_{\text{"27in 4K Gaming Monitor"}} = 12474$  
  $d_{\text{"27in FHD Monitor"}} = 15057$
- $I_i$: Initial Inventory  
  $I_{\text{"27in 4K Gaming Monitor"}} = 62440$  
  $I_{\text{"27in FHD Monitor"}} = 75500$

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**  
Maximize total revenue:
$$
\max \quad 389.99\, x_{\text{"27in 4K Gaming Monitor"}} + 149.99\, x_{\text{"27in FHD Monitor"}}
$$

**Constraints:**  
For each $i \in I$:
1. Demand constraint:
   $$
   x_i \leq d_i
   $$
2. Inventory constraint:
   $$
   x_i \leq I_i
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}
   $$

**Explicitly, the model is:**

$$
\begin{align*}
\max \quad & 389.99\, x_{\text{"27in 4K Gaming Monitor"}} + 149.99\, x_{\text{"27in FHD Monitor"}} \\
\text{s.t.} \quad
& x_{\text{"27in 4K Gaming Monitor"}} \leq 12474 \\
& x_{\text{"27in 4K Gaming Monitor"}} \leq 62440 \\
& x_{\text{"27in FHD Monitor"}} \leq 15057 \\
& x_{\text{"27in FHD Monitor"}} \leq 75500 \\
& x_{\text{"27in 4K Gaming Monitor"}},\ x_{\text{"27in FHD Monitor"}} \in \mathbb{Z}_{\geq 0}
\end{align*}
$$

**Retrieved Information:**

| Product Name                | Revenue | Demand | Initial Inventory |
|-----------------------------|---------|--------|------------------|
| 27in 4K Gaming Monitor      | 389.99  | 12474  | 62440            |
| 27in FHD Monitor            | 149.99  | 15057  | 75500            |