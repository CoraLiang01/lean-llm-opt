##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ to be placed on shelf $i$, for each shelf $i \in S$ and product $j \in P$.

##### Parameters

- $S = \{1,2,3,4,5,6,7,8,9,10\}$ (Shelf IDs)
- $P = \{$
  Smartphone,
  Laptop,
  Headphones,
  Camera,
  Smartwatch,
  Tablet,
  Bluetooth Speaker,
  Keyboard,
  Mouse,
  Monitor,
  Printer,
  External Hard Drive,
  Router,
  Power Bank,
  Memory Card,
  USB Flash Drive,
  Smart Home Hub,
  Gaming Console,
  Fitness Tracker,
  E-Reader
$\}$ (Product Names)

- Shelf capacities:
  - $C_1 = 5.0$
  - $C_2 = 7.0$
  - $C_3 = 6.0$
  - $C_4 = 8.0$
  - $C_5 = 5.5$
  - $C_6 = 9.0$
  - $C_7 = 6.5$
  - $C_8 = 7.5$
  - $C_9 = 8.2$
  - $C_{10} = 5.7$

- Product values and weights:
  - Smartphone: $v_{\text{Smartphone}} = 200$, $w_{\text{Smartphone}} = 1.0$
  - Laptop: $v_{\text{Laptop}} = 1500$, $w_{\text{Laptop}} = 5.0$
  - Headphones: $v_{\text{Headphones}} = 100$, $w_{\text{Headphones}} = 0.5$
  - Camera: $v_{\text{Camera}} = 800$, $w_{\text{Camera}} = 2.0$
  - Smartwatch: $v_{\text{Smartwatch}} = 250$, $w_{\text{Smartwatch}} = 0.3$
  - Tablet: $v_{\text{Tablet}} = 600$, $w_{\text{Tablet}} = 1.5$
  - Bluetooth Speaker: $v_{\text{Bluetooth Speaker}} = 150$, $w_{\text{Bluetooth Speaker}} = 1.0$
  - Keyboard: $v_{\text{Keyboard}} = 80$, $w_{\text{Keyboard}} = 0.8$
  - Mouse: $v_{\text{Mouse}} = 50$, $w_{\text{Mouse}} = 0.2$
  - Monitor: $v_{\text{Monitor}} = 300$, $w_{\text{Monitor}} = 3.0$
  - Printer: $v_{\text{Printer}} = 400$, $w_{\text{Printer}} = 4.0$
  - External Hard Drive: $v_{\text{External Hard Drive}} = 120$, $w_{\text{External Hard Drive}} = 0.5$
  - Router: $v_{\text{Router}} = 60$, $w_{\text{Router}} = 0.3$
  - Power Bank: $v_{\text{Power Bank}} = 40$, $w_{\text{Power Bank}} = 0.4$
  - Memory Card: $v_{\text{Memory Card}} = 30$, $w_{\text{Memory Card}} = 0.05$
  - USB Flash Drive: $v_{\text{USB Flash Drive}} = 25$, $w_{\text{USB Flash Drive}} = 0.02$
  - Smart Home Hub: $v_{\text{Smart Home Hub}} = 100$, $w_{\text{Smart Home Hub}} = 0.6$
  - Gaming Console: $v_{\text{Gaming Console}} = 500$, $w_{\text{Gaming Console}} = 4.0$
  - Fitness Tracker: $v_{\text{Fitness Tracker}} = 90$, $w_{\text{Fitness Tracker}} = 0.2$
  - E-Reader: $v_{\text{E-Reader}} = 180$, $w_{\text{E-Reader}} = 0.5$

##### Objective Function

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. Shelf capacity constraints (for each shelf $i \in S$):

\[
\sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in S
\]

2. Integer and nonnegativity constraints:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
\]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i \in S} \sum_{j \in P} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
\end{align*}
\]

Where all sets, parameters, and coefficients are as listed above.