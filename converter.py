import igraph as ig
import pickle
from pyvis.network import Network

# Correct path and normal pickle loading
with open(r"D:\HippoRAG\HippoRAG\outputs\sample\gpt-4o-mini_text-embedding-3-small\graph.pickle", "rb") as f:
    g = pickle.load(f)

# Create PyVis graph
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")

# Add nodes
for v in g.vs:
    net.add_node(v.index, label=v["name"] if "name" in v.attributes() else str(v.index))

# Add edges
for e in g.es:
    src, tgt = e.tuple
    rel = e["relation"] if "relation" in e.attributes() else ""
    net.add_edge(src, tgt, title=rel)

# Save as HTML
net.show("fastgraphrag_graph.html", notebook=False)


print("✅ Graph visualization saved as fastgraphrag_graph.html")
