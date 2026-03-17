# init_vncore.py
import py_vncorenlp

def get_vncorenlp_model(save_dir='/path/to/vncorenlp'):
    """
    Khởi tạo model với các annotators cần thiết: Tách từ, Từ loại, NER, Cú pháp phụ thuộc.
    """
    model = py_vncorenlp.VnCoreNLP(save_dir=save_dir, annotators=["wseg", "pos", "ner", "parse"])
    return model