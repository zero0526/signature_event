# process.py
from init_vncore import get_vncorenlp_model
from extract_features import ArticleFeatureExtractor
from aggregate_signature import EventSignatureAggregator

def process_event_cluster(articles: list[str]):
    print("1. Khởi tạo mô hình NLP...")
    vncore_model = get_vncorenlp_model('/path/to/vncorenlp')
    extractor = ArticleFeatureExtractor(vncore_model)
    aggregator = EventSignatureAggregator(min_support=0.4) # Ngưỡng 40%

    print(f"2. Bắt đầu trích xuất đặc trưng cho {len(articles)} bài báo...")
    extracted_features_list =[]
    
    for i, text in enumerate(articles):
        features = extractor.extract(text)
        extracted_features_list.append(features)
        print(f"   - Đã xử lý xong bài {i+1}")

    print("3. Tổng hợp và sinh Chữ ký Sự kiện (Event Signature)...")
    event_signature = aggregator.generate_signature(extracted_features_list)

    print("\n=== CHỮ KÝ SỰ KIỆN KHỞI TẠO THÀNH CÔNG ===")
    print(f"- Thực thể chính (Entities/Objects): {event_signature['core_entities']}")
    print(f"- Hành động chính (Actions): {event_signature['core_actions']}")
    
    return event_signature

# TEST CHẠY THỬ
if __name__ == "__main__":
    # Giả lập danh sách 3 bài báo cùng nói về vụ tai nạn
    cluster_articles =[
        "Ngày 17/3, Công an TP Đà Nẵng cho biết đang truy tìm tài xế ô tô 7 chỗ màu đen đã cố tình vượt phải, va chạm với xe máy rồi bỏ chạy.",
        "Mạng xã hội lan truyền clip ô tô Fortuner 7 chỗ lấn làn, đâm xe máy tại Đà Nẵng khiến 2 người ngã xuống đường. Tài xế ô tô đã rời khỏi hiện trường.",
        "Bức xúc vụ ô tô biển vàng vượt ẩu, tông ngã người đi xe máy tại đường Nguyễn Hữu Thọ, Đà Nẵng rồi nhấn ga bỏ chạy."
    ]
    
    process_event_cluster(cluster_articles)