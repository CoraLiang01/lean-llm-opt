##### Sets and Indices
Let $i$ index the set of products classified under ‘Baby’. In this case, there is one product:
- $i$: "Baby Food_255.28"

##### Parameters
- $r_i$: Revenue per unit of product $i$ ("Revenue")
  - $r_{\text{Baby Food\_255.28}} = 255.28$
- $d_i$: Demand for product $i$ ("Demand")
  - $d_{\text{Baby Food\_255.28}} = 765850$
- $s_i$: Initial inventory of product $i$ ("Initial Inventory")
  - $s_{\text{Baby Food\_255.28}} = 5627060$

##### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill demand
  - $x_{\text{Baby Food\_255.28}} \in \mathbb{Z}_{\geq 0}$

##### Objective Function
Maximize total revenue:
$$
\max \; 255.28 \cdot x_{\text{Baby Food\_255.28}}
$$

##### Constraints

1. Inventory constraint:
$$
x_{\text{Baby Food\_255.28}} \leq 5627060
$$

2. Demand constraint:
$$
x_{\text{Baby Food\_255.28}} \leq 765850
$$

3. Non-negativity and integrality:
$$
x_{\text{Baby Food\_255.28}} \in \mathbb{Z}_{\geq 0}
$$

##### Complete Model

$$
\begin{align*}
\max \quad & 255.28 \cdot x_{\text{Baby Food\_255.28}} \\
\text{s.t.} \quad & x_{\text{Baby Food\_255.28}} \leq 5627060 \\
                 & x_{\text{Baby Food\_255.28}} \leq 765850 \\
                 & x_{\text{Baby Food\_255.28}} \in \mathbb{Z}_{\geq 0}
\end{align*}
$$

Where:
- $x_{\text{Baby Food\_255.28}}$ = number of units of "Baby Food_255.28" to fulfill.