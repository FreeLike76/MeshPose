import cv2
import numpy as np
    
from loguru import logger

from time import time
import argparse
from pathlib import Path

from mesh_pose import io
from mesh_pose import retrieval
        
def main(data_p: Path, verbosity: int = 1):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)
    views = data.load_views()
    views_desc = data.load_view_descriptions("SILK", views)
    
    print("Dataset size", len(views_desc))
    
    n_repeat = 1
    
    ts = time()
    for n in range(n_repeat):
        #ret = retrieval.BovwRetrieval(n=0.05)
        ret = retrieval.DLRetrieval(n=0.05, device="cpu")
        #ret = retrieval.PoseRetrieval(n=0.05)
        ret.train(views_desc)
    print("training_time", (time() - ts) / n_repeat)
    
    #for test in range(0, 351, 50):
    #    idx = ret.query(views_desc[test])
    #    for i in idx:
    #        img1 = views_desc[test].view.image
    #        img2 = views_desc[i].view.image
    #        img3 = np.concatenate((img1, img2), axis=1)
    #        img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
    #        cv2.imshow("img", img3)
    #        cv2.waitKey(0)
    
    ts = time()
    repeat = 50
    for n in range(repeat):
        #ret.set_extrinsic(views_desc[n].view.camera.extrinsics)
        idx = ret.query(views_desc[n])
    print(f"Time: {(time() - ts) / repeat * 1000} ms")
    #print("total time", time() - ts)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data\office_model_1", # data\office_model_1
        help="Path to a dataset folder. Default is 'data/first_room/data'.")
    
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