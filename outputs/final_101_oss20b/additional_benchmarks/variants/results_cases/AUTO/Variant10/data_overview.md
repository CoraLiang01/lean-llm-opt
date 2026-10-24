{
  "query": "A utility company must build a least-cost communication backbone connecting all substations in a region. The complete node set is listed in network_nodes.csv, the available undirected links and their construction costs are given in network_edges.csv, and the root node used for the flow-based connectivity formulation is given in network_parameters.csv.\n\nFormulate a minimum spanning tree network-design model. For each candidate link e, define y_e as a binary variable equal to 1 if link e is built and 0 otherwise. To enforce connectivity, define directed auxiliary flow variables f_ij on both directions of each candidate link, sending one unit of flow from the root to every other node. The objective is to minimize total construction cost. The model should select exactly n-1 links, where n is the number of nodes in network_nodes.csv, satisfy the flow-conservation connectivity constraints, link auxiliary flow to selected links, impose nonnegativity on flow variables, and impose binary restrictions on link-selection variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Node"
      ],
      "file_name": "network_nodes.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Node": "N1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Node": "N2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Node": "N3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Node": "N4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Node": "N5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Node": "N6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Node": "N7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Node": "N8"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Node1",
        "Node2",
        "ConstructionCost"
      ],
      "file_name": "network_edges.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "ConstructionCost": "4",
            "Node1": "N1",
            "Node2": "N2"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ConstructionCost": "3",
            "Node1": "N1",
            "Node2": "N3"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ConstructionCost": "9",
            "Node1": "N1",
            "Node2": "N4"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ConstructionCost": "5",
            "Node1": "N2",
            "Node2": "N3"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ConstructionCost": "6",
            "Node1": "N2",
            "Node2": "N5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ConstructionCost": "4",
            "Node1": "N3",
            "Node2": "N4"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ConstructionCost": "7",
            "Node1": "N3",
            "Node2": "N6"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ConstructionCost": "2",
            "Node1": "N4",
            "Node2": "N6"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ConstructionCost": "8",
            "Node1": "N4",
            "Node2": "N7"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ConstructionCost": "3",
            "Node1": "N5",
            "Node2": "N6"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ConstructionCost": "10",
            "Node1": "N5",
            "Node2": "N8"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ConstructionCost": "4",
            "Node1": "N6",
            "Node2": "N7"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ConstructionCost": "6",
            "Node1": "N6",
            "Node2": "N8"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ConstructionCost": "5",
            "Node1": "N7",
            "Node2": "N8"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Parameter",
        "Value"
      ],
      "file_name": "network_parameters.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Parameter": "RootNode",
            "Value": "N1"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
