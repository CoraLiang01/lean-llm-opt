##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j \in J$ to be placed in display area $i \in I$.

##### Parameters

- $I = \{1, 2, 3, \ldots, 14\}$ (Display areas)
- $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (Vessel types)

- Display area capacities:
  - $C_1 = 457$
  - $C_2 = 604$
  - $C_3 = 751$
  - $C_4 = 468$
  - $C_5 = 343$
  - $C_6 = 408$
  - $C_7 = 741$
  - $C_8 = 914$
  - $C_9 = 682$
  - $C_{10} = 409$
  - $C_{11} = 342$
  - $C_{12} = 903$
  - $C_{13} = 680$
  - $C_{14} = 886$

- Vessel values and weights:
  - Speedboat: $v = 29664$, $w = 18$
  - Fishing Boat: $v = 31778$, $w = 36$
  - Catamaran: $v = 73501$, $w = 25$
  - Yacht: $v = 78255$, $w = 16$
  - Sailboat: $v = 93606$, $w = 97$
  - Kayak: $v = 46983$, $w = 35$
  - Canoe: $v = 95026$, $w = 32$
  - Houseboat: $v = 57685$, $w = 100$
  - Pontoon: $v = 60323$, $w = 43$
  - Jet Ski: $v = 91224$, $w = 15$
  - Rowboat: $v = 44003$, $w = 95$
  - Hovercraft: $v = 75998$, $w = 57$
  - Cabin Cruiser: $v = 84525$, $w = 13$
  - Wakeboard Boat: $v = 66207$, $w = 44$
  - Dinghy: $v = 65002$, $w = 64$
  - Trawler: $v = 33132$, $w = 88$
  - Paddle Boat: $v = 69239$, $w = 42$
  - Submarine: $v = 66948$, $w = 46$
  - RIB: $v = 88240$, $w = 24$
  - Skiff: $v = 48858$, $w = 93$

Let $v_j$ and $w_j$ denote the value and weight (size) of vessel type $j$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Capacity constraints for each display area:**
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i \in I} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
\end{align*}
\]

Where all parameters ($C_i$, $v_j$, $w_j$) are as listed above.