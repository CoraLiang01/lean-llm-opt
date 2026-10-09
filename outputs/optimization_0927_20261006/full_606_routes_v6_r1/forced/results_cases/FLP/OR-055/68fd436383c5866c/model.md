##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of boat type $j$ placed in display area $i$, for each display area $i \in I$ and boat type $j \in J$.

##### Parameters

Display areas $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ with capacities:
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

Boat types $J$ with values $v_j$ and sizes $w_j$:
- Speedboat: $v_{\text{Speedboat}} = 69978$, $w_{\text{Speedboat}} = 18$
- Fishing Boat: $v_{\text{Fishing Boat}} = 54011$, $w_{\text{Fishing Boat}} = 42$
- Catamaran: $v_{\text{Catamaran}} = 36352$, $w_{\text{Catamaran}} = 49$
- Yacht: $v_{\text{Yacht}} = 51521$, $w_{\text{Yacht}} = 42$
- Sailboat: $v_{\text{Sailboat}} = 50415$, $w_{\text{Sailboat}} = 41$
- Kayak: $v_{\text{Kayak}} = 76109$, $w_{\text{Kayak}} = 48$
- Canoe: $v_{\text{Canoe}} = 50462$, $w_{\text{Canoe}} = 22$
- Houseboat: $v_{\text{Houseboat}} = 28989$, $w_{\text{Houseboat}} = 29$
- Pontoon: $v_{\text{Pontoon}} = 23318$, $w_{\text{Pontoon}} = 45$
- Jet Ski: $v_{\text{Jet Ski}} = 26142$, $w_{\text{Jet Ski}} = 14$
- Rowboat: $v_{\text{Rowboat}} = 42040$, $w_{\text{Rowboat}} = 38$
- Hovercraft: $v_{\text{Hovercraft}} = 85961$, $w_{\text{Hovercraft}} = 47$
- Cabin Cruiser: $v_{\text{Cabin Cruiser}} = 50142$, $w_{\text{Cabin Cruiser}} = 45$
- Wakeboard Boat: $v_{\text{Wakeboard Boat}} = 48478$, $w_{\text{Wakeboard Boat}} = 28$
- Dinghy: $v_{\text{Dinghy}} = 60953$, $w_{\text{Dinghy}} = 24$
- Trawler: $v_{\text{Trawler}} = 95265$, $w_{\text{Trawler}} = 39$
- Submarine: $v_{\text{Submarine}} = 90957$, $w_{\text{Submarine}} = 36$
- RIB: $v_{\text{RIB}} = 84652$, $w_{\text{RIB}} = 14$
- Skiff: $v_{\text{Skiff}} = 78991$, $w_{\text{Skiff}} = 16$

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

---

###### Retrieved Information

- Display areas $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ with capacities:
  - 1: 356
  - 2: 478
  - 3: 305
  - 4: 291
  - 5: 168
  - 6: 449
  - 7: 139
  - 8: 383
  - 9: 472
  - 10: 288
  - 11: 320
  - 12: 250
  - 13: 402
  - 14: 293

- Boat types $J$ with values and sizes:
  - Speedboat: 69978, 18
  - Fishing Boat: 54011, 42
  - Catamaran: 36352, 49
  - Yacht: 51521, 42
  - Sailboat: 50415, 41
  - Kayak: 76109, 48
  - Canoe: 50462, 22
  - Houseboat: 28989, 29
  - Pontoon: 23318, 45
  - Jet Ski: 26142, 14
  - Rowboat: 42040, 38
  - Hovercraft: 85961, 47
  - Cabin Cruiser: 50142, 45
  - Wakeboard Boat: 48478, 28
  - Dinghy: 60953, 24
  - Trawler: 95265, 39
  - Submarine: 90957, 36
  - RIB: 84652, 14
  - Skiff: 78991, 16