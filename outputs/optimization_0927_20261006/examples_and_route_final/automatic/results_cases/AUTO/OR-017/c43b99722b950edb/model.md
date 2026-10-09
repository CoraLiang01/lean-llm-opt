Let $x_i$ be the number of units of product $i$ (SKU) to fulfill, for each product classified as ‘ZZ’.

#### Sets and Parameters

- $i \in \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}$
- $\text{Revenue}_i$:
  - $\text{ZZ2AO}$: $24.38$
  - $\text{ZZDW7}$: $30.12$
  - $\text{ZZM1A}$: $19.52$
  - $\text{ZZNC5}$: $10.79$
  - $\text{ZZX6K}$: $111.81$
- $\text{Demand}_i$:
  - $\text{ZZ2AO}$: $2$
  - $\text{ZZDW7}$: $4$
  - $\text{ZZM1A}$: $82$
  - $\text{ZZNC5}$: $2$
  - $\text{ZZX6K}$: $2$
- $\text{InitialInventory}_i$:
  - $\text{ZZ2AO}$: $10.0$
  - $\text{ZZDW7}$: $20.0$
  - $\text{ZZM1A}$: $530.0$
  - $\text{ZZNC5}$: $10.0$
  - $\text{ZZX6K}$: $10.0$

#### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i$ (number of units of product $i$ to fulfill)

#### Objective

\[
\max \quad 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}}
\]

#### Constraints

For each $i$:
- Demand constraint: $x_i \leq \text{Demand}_i$
- Inventory constraint: $x_i \leq \text{InitialInventory}_i$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

Explicitly, for each SKU:

\[
\begin{align*}
& x_{\text{ZZ2AO}} \leq 2 \\
& x_{\text{ZZ2AO}} \leq 10.0 \\
& x_{\text{ZZDW7}} \leq 4 \\
& x_{\text{ZZDW7}} \leq 20.0 \\
& x_{\text{ZZM1A}} \leq 82 \\
& x_{\text{ZZM1A}} \leq 530.0 \\
& x_{\text{ZZNC5}} \leq 2 \\
& x_{\text{ZZNC5}} \leq 10.0 \\
& x_{\text{ZZX6K}} \leq 2 \\
& x_{\text{ZZX6K}} \leq 10.0 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]

Where $i \in \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}$.