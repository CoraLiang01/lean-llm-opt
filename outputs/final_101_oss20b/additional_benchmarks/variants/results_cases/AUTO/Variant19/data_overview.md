{
  "query": "A communications team must build a connected backbone over all listed sites. The node set is listed in network_nodes.csv, available undirected links and construction costs are listed in network_edges.csv, and the root node for a single-commodity flow formulation is listed in network_parameters.csv.\n\nFormulate a minimum-cost spanning tree model. For each undirected edge i-j, define y_ij as a binary variable equal to 1 if the edge is selected. For each directed arc version of an available edge, define f_ij as the nonnegative connectivity flow sent from the root. The objective is to minimize selected edge construction cost. The model should include a constraint selecting exactly n-1 edges, root and non-root flow-balance constraints using the node set in network_nodes.csv, flow-to-edge linking constraints with M = n-1, nonnegativity constraints for flow variables, and binary restrictions for edge-selection variables.",
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
            "Node": "V1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Node": "V2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Node": "V3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Node": "V4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Node": "V5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Node": "V6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Node": "V7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Node": "V8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Node": "V9"
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
            "ConstructionCost": "6",
            "Node1": "V1",
            "Node2": "V2"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ConstructionCost": "4",
            "Node1": "V1",
            "Node2": "V3"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ConstructionCost": "9",
            "Node1": "V1",
            "Node2": "V4"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ConstructionCost": "5",
            "Node1": "V2",
            "Node2": "V3"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ConstructionCost": "7",
            "Node1": "V2",
            "Node2": "V5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ConstructionCost": "3",
            "Node1": "V3",
            "Node2": "V4"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ConstructionCost": "8",
            "Node1": "V3",
            "Node2": "V6"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ConstructionCost": "4",
            "Node1": "V4",
            "Node2": "V6"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ConstructionCost": "10",
            "Node1": "V4",
            "Node2": "V7"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ConstructionCost": "6",
            "Node1": "V5",
            "Node2": "V6"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ConstructionCost": "11",
            "Node1": "V5",
            "Node2": "V8"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ConstructionCost": "5",
            "Node1": "V6",
            "Node2": "V7"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ConstructionCost": "7",
            "Node1": "V6",
            "Node2": "V8"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ConstructionCost": "4",
            "Node1": "V7",
            "Node2": "V8"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "ConstructionCost": "6",
            "Node1": "V7",
            "Node2": "V9"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "ConstructionCost": "5",
            "Node1": "V8",
            "Node2": "V9"
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
            "Value": "V1"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
