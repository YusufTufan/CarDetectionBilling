# Gerekli kütüphaneler import edilir
import cv2  # OpenCV ile video işleme yapılacak
import numpy as np  # Sayısal işlemler için NumPy
from ultralytics import YOLO  # YOLOv8 modeli ile nesne tespiti yapılacak
from sort.sort import Sort  # SORT algoritması ile nesne takibi yapılacak
from collections import defaultdict  # Araç sayımı için kullanılır

# YouTube video bağlantısı (araç geçişlerinin olduğu bir köprü videosu)
youtube_url = 'https://www.youtube.com/watch?v=59c6yIYIys8'

#Link-1 : https://www.youtube.com/watch?v=LBjC1JVy9-0
#Link-2 : https://www.youtube.com/watch?v=wqctLW0Hb_0&t=1211s
#Link-3 : https://www.youtube.com/watch?v=59c6yIYIys8 
def get_video_stream(url):
    """
    YouTube URL'sinden mp4 formatında canlı video bağlantısını alır.
    yt_dlp kütüphanesi ile stream bağlantısı çekilir.
    """
    import yt_dlp
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True, 'noplaylist': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        for f in info_dict['formats']:
            if f['vcodec'] != 'none' and f['acodec'] != 'none' and 'url' in f:
                return f['url']
        raise ValueError("Uygun video formatı bulunamadı.")

# Video bağlantısı alınır
video_url = get_video_stream(youtube_url)
print(f"🎥 Video stream bağlantısı: {video_url}")

# YOLOv8 'nano' modeli yüklenir (hafif ve hızlı model)
model = YOLO('yolov8n.pt')

# Sadece şu araç tipleri tespit edilecek
target_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# Her araç sınıfı için geçiş ücreti belirlenir (TL cinsinden)
pricing = {'car': 100, 'motorcycle': 25, 'bus': 150, 'truck': 200, 'bicycle': 0}

# Takip ve sayım için veri yapıları tanımlanır
vehicle_stats = {}  # {id: (sınıf, x, y)} → Her aracın sınıfı ve konumu
vehicle_counts = defaultdict(int)  # Araç türüne göre sayım
total_revenue = 0  # Toplam köprü geliri

# SORT nesne takip algoritması başlatılır
tracker = Sort()

# OpenCV ile video akışı başlatılır
cap = cv2.VideoCapture(video_url)

# Video açılamazsa hata ver ve çık
if not cap.isOpened():
    print("❌ Video açılamadı.")
    input("Program tamamlandı. Kapatmak için Enter'a bas...")
    exit()
else:
    print("✅ Video başarıyla açıldı.")

# FPS bilgisi alınır ve belirli bir zaman (1132. saniye) atlanır
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(fps * 1132)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Görüntü penceresi oluşturulur
cv2.namedWindow("Arac_Takip_Kopru_Ucreti", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Arac_Takip_Kopru_Ucreti", 960, 540)

# FPS kontrolü yapılır (her 20 karede bir analiz yapılacak)
fps_control = 0
rate_limit = 20

def is_duplicate_vehicle(new_center, new_class, threshold=50):
    """
    Daha önce aynı türden ve benzer konumda bir araç geçtiyse True döner.
    Bu sayede tekrar sayım önlenir.
    """
    for _, (v_class, cx, cy) in vehicle_stats.items():
        if v_class == new_class:
            dist = np.linalg.norm(np.array(new_center) - np.array((cx, cy)))
            if dist < threshold:
                return True
    return False

# Ana döngü başlatılır: her kare okunur ve işlenir
while True:
    fps_control += 1
    if fps_control % rate_limit != 0:
        continue  # Her 20 karede bir analiz yap

    ret, frame = cap.read()
    if not ret:
        break  # Video sonlandıysa çık

    frame = cv2.resize(frame, (640, 360))  # Daha hızlı işlem için küçültülür

    # YOLO modeli ile nesne tespiti yapılır
    results = model(frame)
    detections = []  # Tespit edilen kutular
    class_map = {}   # Kutunun hangi sınıfa ait olduğu

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Kutu koordinatları
            conf = box.conf[0].item()  # Güven skorları
            cls = int(box.cls[0].item())  # Sınıf numarası
            class_name = model.names[cls]  # Sınıf adı alınır
            if class_name in target_classes:
                detections.append([x1, y1, x2, y2, conf])  # Kutu eklenir
                class_map[(x1, y1, x2, y2)] = class_name  # Sınıfı eşle

    # Tespit edilen kutular SORT algoritmasına verilir
    if detections:
        det_array = np.array(detections, dtype=np.float32)
        tracked_objects = tracker.update(det_array)
    else:
        tracked_objects = np.empty((0, 5))

    # Takip edilen her nesne işlenir
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        matched_class = None
        best_iou = 0  # En iyi eşleşme için IoU kullanılacak

        # Takip kutusunu tespit kutusu ile eşle (IoU kullanılarak)
        for box_coords, class_name in class_map.items():
            iou_x1, iou_y1, iou_x2, iou_y2 = box_coords
            xi1 = max(iou_x1, x1)
            yi1 = max(iou_y1, y1)
            xi2 = min(iou_x2, x2)
            yi2 = min(iou_y2, y2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (iou_x2 - iou_x1) * (iou_y2 - iou_y1)
            box2_area = (x2 - x1) * (y2 - y1)
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area != 0 else 0

            if iou > best_iou:
                best_iou = iou
                matched_class = class_name

        # Eşleşen araç varsa işleme devam
        if matched_class:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Daha önce sayılmamışsa ücret ve sayım yapılır
            if obj_id not in vehicle_stats:
                if not is_duplicate_vehicle((center_x, center_y), matched_class):
                    vehicle_stats[obj_id] = (matched_class, center_x, center_y)
                    vehicle_counts[matched_class] += 1
                    total_revenue += pricing[matched_class]

            # Araç kutusu ve sınıf adı çizilir
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, matched_class, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Üst bilgi paneli (yarı saydam alan)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (960, 40), (50, 50, 50), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Özet bilgi yazısı: toplam gelir ve araç sayısı
    summary = f"Gelir: {total_revenue} TL | " + " | ".join(
        [f"{k}: {vehicle_counts[k]}" for k in target_classes]
    )
    cv2.putText(frame, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Görüntü ekranda gösterilir
    cv2.imshow("Arac_Takip_Kopru_Ucreti", frame)

    # Çıkış tuşu: 'q' basılırsa döngü durur
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video akışı ve pencere kapatılır
cap.release()
cv2.destroyAllWindows()
