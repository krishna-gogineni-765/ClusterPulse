class PulseCluster:
    def __init__(self, cluster_id=None):
        self.cluster_id = cluster_id
        self.documents = []
        self.cluster_centroid = None
        self.cluster_label = None
        self.cluster_description = None

    def add_document(self, document):
        self.documents.append(document)

    def get_representative_points(self, num_points=15):
    #TODO : Return represenatative points based on probability
        return self.documents[:num_points]

