import networkx as nx

# Create a new empty graph
G = nx.Graph()

# Add some nodes and edges
G.add_node("Alice", type="person")
G.add_node("Bob", type="person")
G.add_node("Company X", type="company")
G.add_node("Engineer", type="job_title")
G.add_edge("Alice", "Engineer", role="employee")
G.add_edge("Engineer", "Company X", role="works_for")
G.add_edge("Bob", "Engineer", role="friend")

# Analyze the graph
print(nx.shortest_path(G, "Alice", "Company X"))
print(nx.closeness_centrality(G, "Engineer"))
