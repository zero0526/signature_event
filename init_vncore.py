import py_vncorenlp
import os

def get_vncorenlp_model(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        py_vncorenlp.download_model(save_dir=os.path.abspath(save_dir))
    return py_vncorenlp.VnCoreNLP(save_dir=os.path.abspath(save_dir))