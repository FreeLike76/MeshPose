
from .project import ProjectMeta
from .features import extractors

class Localization:
    def __init__(self, project_meta: ProjectMeta, verbose: bool = False):
        self.project_meta = project_meta
        feature_extractor_meta = project_meta.get_processed_features_meta_init()
        self.feature_extractor = extractors.DEFINED_EXTRACTORS[feature_extractor_meta["class"]].from_json(feature_extractor_meta["params"])
        
    """
    def init
        - load ProjectMeta
        
        - set parameters
        - create algorithms
    
    def setup
        - set data
    
    def run: image
        - run localization on image
        - return pose
    """