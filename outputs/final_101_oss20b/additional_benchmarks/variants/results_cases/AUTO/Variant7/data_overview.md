{
  "query": "A distribution network ships goods from source plants to customers through intermediate cross-dock hubs. Source supplies and customer demands are listed in node_supply_demand.csv, hub throughput capacities are listed in hub_capacity.csv, and per-unit transportation costs for source-to-hub and hub-to-customer arcs are listed in arc_costs.csv. The total shipments out of each source may not exceed its available supply; unused source supply is allowed.\n\nFormulate a minimum-cost transshipment model. For each directed arc i-j, define f_ij as the nonnegative shipment flow on that arc. The objective is to minimize total transportation cost. The model should include source supply upper-bound constraints, customer demand constraints, flow-balance constraints at each hub, hub throughput-capacity constraints, and nonnegativity constraints for all arc-flow variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Node",
        "NodeType",
        "Amount"
      ],
      "file_name": "node_supply_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Amount": "120",
            "Node": "S1",
            "NodeType": "SourceSupply"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Amount": "100",
            "Node": "S2",
            "NodeType": "SourceSupply"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Amount": "90",
            "Node": "S3",
            "NodeType": "SourceSupply"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Amount": "70",
            "Node": "C1",
            "NodeType": "CustomerDemand"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Amount": "80",
            "Node": "C2",
            "NodeType": "CustomerDemand"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Amount": "60",
            "Node": "C3",
            "NodeType": "CustomerDemand"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Amount": "90",
            "Node": "C4",
            "NodeType": "CustomerDemand"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Hub",
        "ThroughputCapacity"
      ],
      "file_name": "hub_capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Hub": "H1",
            "ThroughputCapacity": "170"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Hub": "H2",
            "ThroughputCapacity": "160"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "From",
        "To",
        "Cost"
      ],
      "file_name": "arc_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Cost": "2",
            "From": "S1",
            "To": "H1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Cost": "6",
            "From": "S1",
            "To": "H2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Cost": "4",
            "From": "S2",
            "To": "H1"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Cost": "3",
            "From": "S2",
            "To": "H2"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Cost": "7",
            "From": "S3",
            "To": "H1"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Cost": "2",
            "From": "S3",
            "To": "H2"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Cost": "3",
            "From": "H1",
            "To": "C1"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Cost": "4",
            "From": "H1",
            "To": "C2"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Cost": "7",
            "From": "H1",
            "To": "C3"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Cost": "8",
            "From": "H1",
            "To": "C4"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Cost": "8",
            "From": "H2",
            "To": "C1"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Cost": "6",
            "From": "H2",
            "To": "C2"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "Cost": "3",
            "From": "H2",
            "To": "C3"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "Cost": "4",
            "From": "H2",
            "To": "C4"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
