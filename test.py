from loguru import logger

import argparse
from pathlib import Path

from mesh_pose import io
from mesh_pose.retrieval import DLRetrieval
        
def main(data_p: Path, verbosity: int = 1):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)
    views = data.load_views()
    views_desc = data.load_view_descriptions("ORB", views)
    
    ret = DLRetrieval()
    ret.train(views_desc)
    
    from time import time
    
    ts = time()
    repeat = 50
    for n in range(repeat):
        idx = ret.query(views_desc[n].view)
    print(f"Time: {(time() - ts) / repeat * 1000} ms")
    print("total time", time() - ts)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/second_room/data",
        help="Path to a dataset folder. Default is 'data/second_room/data'.")
    
    parser.add_argument(
        "--verbosity", type=int, choices=[0, 1, 2], default=1)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    
    # Get params
    data_p = Path(args.data)
    verbosity = args.verbosity
    
    # Verify path
    assert data_p.exists() and data_p.is_dir(), logger.error(
        f"Directory {str(data_p)} does not exist!")
    
    # Run main
    main(data_p, verbosity=verbosity)