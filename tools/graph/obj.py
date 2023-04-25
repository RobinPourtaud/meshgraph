import networkx as nx
import torch

class ObjModel:
    """An ObjModel is an object based on the data in an .obj file.

    Attributes:
        filename: A string representing the path to the .obj file.
        label: A string representing the label of the image.
        vertices: A list of vertices.
        faces: A list of faces.
        normals: A list of normals.
        texcoords: A list of texture coordinates.
        material: A string representing the material.
        networkx_node_graph: A networkx graph object based on the vertices and edges.
        networkx_face_graph: A networkx graph object based on the faces as vertices and adjacent faces as edges.
        networkx_simplex_graph: A networkx graph object based on the simplices as vertices and adjacent simplices as edges.
    """

    def __init__(self, filename, label=None, load=True):
        """Initialize an ObjModel object.

        Args:
            filename: A string representing the path to the .obj file.
            label: (Optional) A string representing the label of the image.
            load: (Optional) A boolean indicating whether to load the model from the obj file. Default is True.
        """
        self.filename = filename
        self.label = label
        self.vertices = {
            'x': [],
            'y': [],
            'z': [], 
            'normal': [],
        }
        self.faces = set()
        self.edges = set()

        self.networkx_node_graph = None
        self.networkx_face_graph = None
        self.networkx_simplex_graph = None

        if load:
            self._load_obj_file()

    def _load_obj_file(self):
        """Loads the model from the obj file."""
        with open(self.filename, "r") as f:
            for line in f:
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    v = list(map(float, values[1:4]))
                    self.vertices['x'].append(v[0])
                    self.vertices['y'].append(v[1])
                    self.vertices['z'].append(v[2])
                elif values[0] == 'vn':
                    n = list(map(float, values[1:4]))
                    self.vertices['normal'].append(n)
                elif values[0] == 'f':
                    face = []
                    for v in values[1:]:
                        w = v.split('//')
                        face.append(int(w[0]))
                    self.faces.add(tuple(face))

                    # Add edges
                    face_edges = {(face[i] - 1, face[(i + 1) % len(face)] - 1) for i in range(len(face))}
                    self.edges.update(face_edges)


    def create_networkx_node_graph(self, inplace=True):
        """Create a networkx graph with only nodes and edges based on the original vertices.

        Args:
            inplace: (Optional) A boolean indicating whether to update the object attribute. Default is True.

        Returns:
            A networkx graph object.
        """
        node_graph = nx.Graph()

        # Add nodes to the graph
        for idx, (x, y, z) in enumerate(zip(self.vertices['x'], self.vertices['y'], self.vertices['z'])):
            node_graph.add_node(idx, vertex=(x, y, z))

        # Add edges to the graph
        for edge in self.edges:
            node_graph.add_edge(*edge)

        if inplace:
            self.networkx_node_graph = node_graph

        return node_graph

    def create_networkx_face_graph(self, inplace=True):
        """Create a networkx graph with faces as vertices and adjacent faces as edges.

        Args:
            inplace: (Optional) A boolean indicating whether to update the object attribute. Default is True.

        Returns:
            A networkx graph object.
        """
        face_graph = nx.Graph()

        # Add faces as nodes to the graph
        for idx, face in enumerate(self.faces):
            face_graph.add_node(idx, face=face)

        # Add edges between adjacent faces
        for idx1, face1 in enumerate(self.faces):
            for idx2, face2 in enumerate(self.faces):
                if idx1 != idx2 and set(face1) & set(face2):
                    face_graph.add_edge(idx1, idx2)

        if inplace:
            self.networkx_face_graph = face_graph

        return face_graph

    def to_tensor(self, feature=None):
        """Convert the networkx node graph to a PyTorch tensor.

        Args:
            feature: (Optional) A string representing the node feature to use for tensor creation.

        Returns:
            A tuple containing the adjacency tensor and the feature tensor (if specified).
        """
        import numpy as np
        assert isinstance(self.networkx_node_graph, nx.Graph), "Node graph must be a networkx graph object"

        adjacency_matrix = nx.adjacency_matrix(self.networkx_node_graph).toarray()
        adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float)

        if feature:
            feature_matrix = np.array([self.networkx_node_graph.nodes[node][feature] for node in self.networkx_node_graph.nodes])
            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float)
            return adjacency_tensor, feature_tensor

        return adjacency_tensor,

        
        
    