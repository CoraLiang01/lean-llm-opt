##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of boat type $j$ placed in display area $i$, for each display area $i \in I$ and boat type $j \in J$.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ (Display area indices)
- $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (Boat types)

- Display area capacities $C_i$:
  - $C_1 = 356$
  - $C_2 = 478$
  - $C_3 = 305$
  - $C_4 = 291$
  - $C_5 = 168$
  - $C_6 = 449$
  - $C_7 = 139$
  - $C_8 = 383$
  - $C_9 = 472$
  - $C_{10} = 288$
  - $C_{11} = 320$
  - $C_{12} = 250$
  - $C_{13} = 402$
  - $C_{14} = 293$

- Boat type values $v_j$ and sizes $w_j$:
  - Speedboat: $v_1 = 69978$, $w_1 = 18$
  - Fishing Boat: $v_2 = 54011$, $w_2 = 42$
  - Catamaran: $v_3 = 36352$, $w_3 = 49$
  - Yacht: $v_4 = 51521$, $w_4 = 42$
  - Sailboat: $v_5 = 50415$, $w_5 = 41$
  - Kayak: $v_6 = 76109$, $w_6 = 48$
  - Canoe: $v_7 = 50462$, $w_7 = 22$
  - Houseboat: $v_8 = 28989$, $w_8 = 29$
  - Pontoon: $v_9 = 23318$, $w_9 = 45$
  - Jet Ski: $v_{10} = 26142$, $w_{10} = 14$
  - Rowboat: $v_{11} = 42040$, $w_{11} = 38$
  - Hovercraft: $v_{12} = 85961$, $w_{12} = 47$
  - Cabin Cruiser: $v_{13} = 50142$, $w_{13} = 45$
  - Wakeboard Boat: $v_{14} = 48478$, $w_{14} = 28$
  - Dinghy: $v_{15} = 60953$, $w_{15} = 24$
  - Trawler: $v_{16} = 95265$, $w_{16} = 39$
  - Paddle Boat: $v_{17} = 22839$, $w_{17} = 32$
  - Submarine: $v_{18} = 90957$, $w_{18} = 36$
  - RIB: $v_{19} = 84652$, $w_{19} = 14$
  - Skiff: $v_{20} = 78991$, $w_{20} = 16$

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
\max\quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,14;\ j=1,\ldots,20
\end{align*}
\]

Where all $v_j$, $w_j$, and $C_i$ are as listed above.