import networkx as nx

class GraphBuilder:
    """
    Constructs a NetworkX graph from stream events.
    """
    def __init__(self, id2entity, id2relation):
        self.id2entity = id2entity
        self.id2relation = id2relation
        
    def build_from_events(self, events):
        """
        events: List of dicts {'token_step': int, 'entity': int, 'relation': int}
        """
        G = nx.DiGraph()
        
        # Simple heuristic: Sequential triples?
        # Or just bag of entities/relations?
        # For now, let's assume events trigger node creation or edge creation
        # This logic heavily depends on the probe training targets (BIO tags vs. direct classification)
        
        # Placeholder logic assuming "Active" entity prediction
        for event in events:
            ent_label = self.id2entity.get(event['entity'], "O")
            rel_label = self.id2relation.get(event['relation'], "O")
            
            if ent_label != "O":
                G.add_node(event['token_step'], label=ent_label)
                
            if rel_label != "O":
                 G.add_edge("CurrentContext", event['token_step'], label=rel_label)
                 
        return G
