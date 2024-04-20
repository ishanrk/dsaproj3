"""
{"<index : int>" : {"<region : str>": int, "<neuron_size>" : int, "<connection_bias>" : float, "<error_bias>" : float, "<adjacency_list>" : list[int]}

- Where <> refers to the name of the key in JSON followed by its type.

Example:

{"123" : {"region": 3, "neuron_size" : 1, "connection_bias" : 0.126, "error_bias" : 0.02, "adjacency_list" : [4, 2, 0]}

If the specified format not provided, provide a dictionary that maps the standard specified above
to the format used in JSON.

Example:

property_mapper = {"neuron_size" : "size", "adjacency_list" : "connections"}
"""

ishan_to_std_json = {"region" : "location", "neuron_size" : "neuron_size", "connection_bias" : "connection_bias", "error_bias" : "error", "adjacency_list" : "connections"}
