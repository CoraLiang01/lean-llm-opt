{
  "query": "A mobile service program must choose facility locations to serve residential areas. Area demand is listed in area_demand.csv, distances from candidate locations to areas are listed in location_distance.csv, and the number of facilities to open is listed in planning_parameters.csv. Each area must be assigned to exactly one opened facility.\n\nFormulate a minimum-demand-weighted-distance p-median model. For each candidate location i, define y_i as a binary variable equal to 1 if location i is opened. For each location-area pair i-j, define x_ij as a binary variable equal to 1 if area j is assigned to location i. The objective is to minimize total demand-weighted assignment distance. The model should include exactly-one assignment constraints for all areas, a constraint opening exactly the required number of facilities, assignment-to-open-facility linking constraints, and binary restrictions for all variables.",
  "relationships": [
    {
      "column_axis": {
        "id_column": "Area",
        "table_id": "file_0_view_0"
      },
      "matrix_table_id": "file_1_view_0",
      "row_axis": {
        "id_column": "Location",
        "table_id": "file_1_view_0"
      },
      "row_id_column": "Location",
      "type": "matrix"
    }
  ],
  "tables": [
    {
      "columns": [
        "Area",
        "Demand"
      ],
      "file_name": "area_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Area": "A1",
            "Demand": "25"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Area": "A2",
            "Demand": "35"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Area": "A3",
            "Demand": "40"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Area": "A4",
            "Demand": "30"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Area": "A5",
            "Demand": "50"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Area": "A6",
            "Demand": "45"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Area": "A7",
            "Demand": "20"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Area": "A8",
            "Demand": "55"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Area": "A9",
            "Demand": "60"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Area": "A10",
            "Demand": "30"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Area": "A11",
            "Demand": "42"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Area": "A12",
            "Demand": "38"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Location",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12"
      ],
      "file_name": "location_distance.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "A1": "2",
            "A10": "12",
            "A11": "11",
            "A12": "10",
            "A2": "3",
            "A3": "4",
            "A4": "8",
            "A5": "9",
            "A6": "10",
            "A7": "13",
            "A8": "14",
            "A9": "15",
            "Location": "L1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "A1": "3",
            "A10": "11",
            "A11": "10",
            "A12": "9",
            "A2": "2",
            "A3": "3",
            "A4": "7",
            "A5": "8",
            "A6": "9",
            "A7": "12",
            "A8": "13",
            "A9": "14",
            "Location": "L2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "A1": "8",
            "A10": "7",
            "A11": "6",
            "A12": "7",
            "A2": "7",
            "A3": "5",
            "A4": "2",
            "A5": "3",
            "A6": "4",
            "A7": "8",
            "A8": "9",
            "A9": "11",
            "Location": "L3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "A1": "9",
            "A10": "6",
            "A11": "5",
            "A12": "6",
            "A2": "8",
            "A3": "6",
            "A4": "3",
            "A5": "2",
            "A6": "3",
            "A7": "7",
            "A8": "8",
            "A9": "10",
            "Location": "L4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "A1": "13",
            "A10": "5",
            "A11": "6",
            "A12": "7",
            "A2": "12",
            "A3": "10",
            "A4": "8",
            "A5": "7",
            "A6": "6",
            "A7": "2",
            "A8": "3",
            "A9": "4",
            "Location": "L5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "A1": "14",
            "A10": "4",
            "A11": "5",
            "A12": "6",
            "A2": "13",
            "A3": "11",
            "A4": "9",
            "A5": "8",
            "A6": "7",
            "A7": "3",
            "A8": "2",
            "A9": "3",
            "Location": "L6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "A1": "11",
            "A10": "2",
            "A11": "3",
            "A12": "2",
            "A2": "10",
            "A3": "8",
            "A4": "7",
            "A5": "6",
            "A6": "5",
            "A7": "6",
            "A8": "5",
            "A9": "4",
            "Location": "L7"
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
      "file_name": "planning_parameters.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Parameter": "NumberOfFacilitiesToOpen",
            "Value": "3"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
