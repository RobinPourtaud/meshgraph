from .objShape import ObjModel


class ObjCollection:    
    """A class to handle all object in ShapenetCorev2 dataset.

    It provides some neet methods to load and manipulates the different models

    Attributes:
        models (dict): A dictionary of obj files. Keys are the labels of the labels, values are lists of ObjModel objects.
    """

    def __init__(self, autoload : bool = True): 
        """Initialize an ObjCollection object.

        Args:
            autoload (bool, optional): If True, load all the obj files. Defaults to True.

        
        """
        self.models = {}
        self.categories = {} 
        self.taxonomy = {}
        self.materials = {}
        self.metadata = {}
        
        if autoload:
            self.load_all()


    def load_all(self):
        """Load all the obj files in the dataset
        """
        import json
        with open('models/taxonomy.json') as f:
            data = json.load(f)

        labels = {}
        for i in range(len(data)):
            labels[data[i]['synsetId']] = data[i]['name']

        import os
        for folder in os.listdir('models/'):
            # if folder is a folder
            if os.path.isdir('models/' + folder):
                self.models[labels[folder]] = []
                for subfolder in os.listdir('models/' + folder):
                    if os.path.isdir('models/' + folder + '/' + subfolder):
                        for file in os.listdir('models/' + folder + '/' + subfolder):
                            if file.endswith('.obj'):
                                
                                self.models[labels[folder]].append(ObjModel('models/' + folder + '/' + subfolder + '/' + file, labels[folder]))
                            if file == "models": 
                                for file2 in os.listdir('models/' + folder + '/' + subfolder + '/' + file):
                                    if file2.endswith('.obj'):
                                        self.models[labels[folder]].append(ObjModel('models/' + folder + '/' + subfolder + '/' + file + '/' + file2, labels[folder]))

        