def make_passthrough_node(key):
 def node(state):return {key:state.get(key)}
 return node
