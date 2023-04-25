import networkx as nx

class ObjModel:
    def __init__(self, file: str):
        self.file = file
        self.networkx_obj = None

    def to_networkx(self, multigraph: bool = False):
        # Initialize an empty graph or multigraph based on the parameter
        self.networkx_obj = nx.MultiGraph() if multigraph else nx.Graph()

        # Read the file
        with open(self.file, 'r') as file:
            normals = {}
            for line in file:
                # Split the line into parts
                parts = line.split()

                # Skip if the line is empty
                if not parts:
                    continue

                # If it's a vertex normal, store it
                if parts[0] == 'vn':
                    normal_id = len(normals) + 1
                    normals[normal_id] = (float(parts[1]), float(parts[2]), float(parts[3]))

                # If it's a vertex, add it as a node
                elif parts[0] == 'v':
                    node_id = len(self.networkx_obj.nodes) + 1
                    self.networkx_obj.add_node(node_id, coordinates=(float(parts[1]), float(parts[2]), float(parts[3])))

                # If it's a face, add it as an edge
                elif parts[0] == 'f':
                    for i in range(1, len(parts) - 1):
                        face_data = parts[i].replace('//', '/').split('/')
                        node1, _, normal_id1 = map(int, face_data) if len(face_data) == 3 else (int(face_data[0]), None, int(face_data[1]))

                        face_data = parts[i + 1].replace('//', '/').split('/')
                        node2, _, normal_id2 = map(int, face_data) if len(face_data) == 3 else (int(face_data[0]), None, int(face_data[1]))

                        normal_id = (normal_id1 + normal_id2) // 2  # Use the average of both normal ids
                        self.networkx_obj.add_edge(node1, node2, face_normal=normals[normal_id])

        return self.networkx_obj
