##### Sets

Let $I = \{1,2,\ldots,14\}$ be the set of display areas (DisplayID).
Let $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ be the set of boat types.

##### Parameters

Display area capacities (from capacity.csv):

\[
\begin{align*}
C_1 &= 356 \\
C_2 &= 478 \\
C_3 &= 305 \\
C_4 &= 291 \\
C_5 &= 168 \\
C_6 &= 449 \\
C_7 &= 139 \\
C_8 &= 383 \\
C_9 &= 472 \\
C_{10} &= 288 \\
C_{11} &= 320 \\
C_{12} &= 250 \\
C_{13} &= 402 \\
C_{14} &= 293 \\
\end{align*}
\]

Boat type values and sizes (from products.csv):

| $j$              | Value $v_j$ | Size $w_j$ |
|------------------|-------------|------------|
| Speedboat        | 69978       | 18         |
| Fishing Boat     | 54011       | 42         |
| Catamaran        | 36352       | 49         |
| Yacht            | 51521       | 42         |
| Sailboat         | 50415       | 41         |
| Kayak            | 76109       | 48         |
| Canoe            | 50462       | 22         |
| Houseboat        | 28989       | 29         |
| Pontoon          | 23318       | 45         |
| Jet Ski          | 26142       | 14         |
| Rowboat          | 42040       | 38         |
| Hovercraft       | 85961       | 47         |
| Cabin Cruiser    | 50142       | 45         |
| Wakeboard Boat   | 48478       | 28         |
| Dinghy           | 60953       | 24         |
| Trawler          | 95265       | 39         |
| Paddle Boat      | 22839       | 32         |
| Submarine        | 90957       | 36         |
| RIB              | 84652       | 14         |
| Skiff            | 78991       | 16         |

##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of boat type $j \in J$ to be placed in display area $i \in I$.

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

##### Parameters (explicit listing)

- $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$
- $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$
- $C = [356, 478, 305, 291, 168, 449, 139, 383, 472, 288, 320, 250, 402, 293]$
- $v = [69978, 54011, 36352, 51521, 50415, 76109, 50462, 28989, 23318, 26142, 42040, 85961, 50142, 48478, 60953, 95265, 22839, 90957, 84652, 78991]$
- $w = [18, 42, 49, 42, 41, 48, 22, 29, 45, 14, 38, 47, 45, 28, 24, 39, 32, 36, 14, 16]$