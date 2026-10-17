Let $x_{ij}$ be the number of vessels of type $j$ (ProductName $j$) to be placed in display area $i$ (DisplayID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of vessel type $j$ (from Value column in products.csv)
- $W_j$: Weight (size) of vessel type $j$ (from Weight column in products.csv)
- $C_i$: Capacity of display area $i$ (from Capacity column in capacity.csv)

**Sets:**

- $i \in \{1,2,\ldots,14\}$ (DisplayID from capacity.csv, in given order)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv, in given order)

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{14} \sum_{j} V_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i$ (DisplayID):

\[
\sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,14\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameter Values (in source order):**

- Display areas (DisplayID) and capacities:
    - 1: 457
    - 2: 604
    - 3: 751
    - 4: 468
    - 5: 343
    - 6: 408
    - 7: 741
    - 8: 914
    - 9: 682
    - 10: 409
    - 11: 342
    - 12: 903
    - 13: 680
    - 14: 886

- Vessel types (ProductName), values, and weights:
    - Speedboat: Value 29664, Weight 18
    - Fishing Boat: Value 31778, Weight 36
    - Catamaran: Value 73501, Weight 25
    - Yacht: Value 78255, Weight 16
    - Sailboat: Value 93606, Weight 97
    - Kayak: Value 46983, Weight 35
    - Canoe: Value 95026, Weight 32
    - Houseboat: Value 57685, Weight 100
    - Pontoon: Value 60323, Weight 43
    - Jet Ski: Value 91224, Weight 15
    - Rowboat: Value 44003, Weight 95
    - Hovercraft: Value 75998, Weight 57
    - Cabin Cruiser: Value 84525, Weight 13
    - Wakeboard Boat: Value 66207, Weight 44
    - Dinghy: Value 65002, Weight 64
    - Trawler: Value 33132, Weight 88
    - Paddle Boat: Value 69239, Weight 42
    - Submarine: Value 66948, Weight 46
    - RIB: Value 88240, Weight 24
    - Skiff: Value 48858, Weight 93

---

**Decision variables:**

\[
x_{ij} = \text{number of vessels of type $j$ to be placed in display area $i$}, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]