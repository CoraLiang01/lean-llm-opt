##### Objective Function: 

$ \quad \quad \max \sum_{i} \sum_{j} p_{j} \cdot x_{ij} $

##### Constraints

###### 1. Capacity Constraints: 

$ \quad \quad \sum_{j} a_{j} \, x_{ij} \leq c_{i}, \quad \forall i $

###### 2. Non‑negativity and Integrality Constraints: 

$ \quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i,j $

Here $i$ indexes the display areas (DisplayID), $j$ indexes the vessel types (ProductName), $p_{j}$ is the Value of vessel $j$, $a_{j}$ is the Weight of vessel $j$, and $c_{i}$ is the Capacity of display area $i$. All variables represent whole vessels.

###### Retrieved Information

{
  "items": [
    "Speedboat",
    "Fishing Boat",
    "Catamaran",
    "Yacht",
    "Sailboat",
    "Kayak",
    "Canoe",
    "Houseboat",
    "Pontoon",
    "Jet Ski",
    "Rowboat",
    "Hovercraft",
    "Cabin Cruiser",
    "Wakeboard Boat",
    "Dinghy",
    "Trawler",
    "Paddle Boat",
    "Submarine",
    "RIB",
    "Skiff"
  ],
  "profit": {
    "Speedboat": 29664,
    "Fishing Boat": 31778,
    "Catamaran": 73501,
    "Yacht": 78255,
    "Sailboat": 93606,
    "Kayak": 46983,
    "Canoe": 95026,
    "Houseboat": 57685,
    "Pontoon": 60323,
    "Jet Ski": 91224,
    "Rowboat": 44003,
    "Hovercraft": 75998,
    "Cabin Cruiser": 84525,
    "Wakeboard Boat": 66207,
    "Dinghy": 65002,
    "Trawler": 33132,
    "Paddle Boat": 69239,
    "Submarine": 66948,
    "RIB": 88240,
    "Skiff": 48858
  },
  "shelves": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14"
  ],
  "capacity": {
    "1": 457,
    "2": 604,
    "3": 751,
    "4": 468,
    "5": 343,
    "6": 408,
    "7": 741,
    "8": 914,
    "9": 682,
    "10": 409,
    "11": 342,
    "12": 903,
    "13": 680,
    "14": 886
  },
  "space": {
    "Speedboat": 18,
    "Fishing Boat": 36,
    "Catamaran": 25,
    "Yacht": 16,
    "Sailboat": 97,
    "Kayak": 35,
    "Canoe": 32,
    "Houseboat": 100,
    "Pontoon": 43,
    "Jet Ski": 15,
    "Rowboat": 95,
    "Hovercraft": 57,
    "Cabin Cruiser": 13,
    "Wakeboard Boat": 44,
    "Dinghy": 64,
    "Trawler": 88,
    "Paddle Boat": 42,
    "Submarine": 46,
    "RIB": 24,
    "Skiff": 93
  }
}