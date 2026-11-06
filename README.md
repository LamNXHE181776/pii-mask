# Dịch vụ che thông tin cá nhân trên ảnh giấy tờ

API FastAPI nhận ảnh giấy tờ (căn cước, hộ chiếu, chứng từ y tế/bảo hiểm…), tìm
mọi thông tin cá nhân, tô đè lên ảnh rồi trả về ảnh đã che cùng danh sách các
vùng đã che.

Một ảnh vào, một JSON ra:

```json
{
  "width": 929,
  "height": 580,
  "regions": [
    {"category": "person_name", "polygon": [[340.0,122.0],[400.0,122.0],[400.0,153.0],[340.0,153.0]], "text": "TUUL"},
    {"category": "face", "polygon": [[32.0,92.0],[323.0,92.0],[323.0,377.0],[32.0,377.0]], "text": null}
  ],
  "masked_image": "/9j/4AAQSkZJRg...",
  "processing_time_ms": 761
}
```

> Bản tiếng Anh: [README.en.md](README.en.md)

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Driver NVIDIA | ≥ 580 (CUDA 13.x) |
| Docker | ≥ 24 |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | mới nhất |
| VRAM | ≥ 4 GB (thực tế dùng dưới 2 GB) |
| Python (chỉ khi chạy local) | 3.12 |
| [uv](https://docs.astral.sh/uv/) (chỉ khi chạy local) | mới nhất |

Cấu hình GPU runtime trên host một lần duy nhất:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Chạy nhanh bằng Docker

```bash
docker build -t pii-mask:1 .
docker run --rm --gpus all -p 8000:8000 pii-mask:1
```

Sau đó:

- **http://127.0.0.1:8000/docs** — giao diện Swagger
- **http://127.0.0.1:8000/health** — kiểm tra dịch vụ còn sống

```bash
curl -s -X POST http://127.0.0.1:8000/v1/mask \
     -F "file=@candidate/images/idnet_00003.jpg" | head -c 400
```

Image hoàn toàn tự chứa và chạy được **không cần internet**:

```bash
docker run --rm --gpus all --network none pii-mask:1
```

## Chạy khi phát triển

```bash
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Thư mục `models/` phải có đủ ba tệp nêu bên dưới. `pipeline.py` xác định đường dẫn
theo vị trí của chính nó, nên chạy từ thư mục nào cũng được.

---

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /v1/mask`

`multipart/form-data`, đúng một trường tên **`file`** (JPEG hoặc PNG).

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `width`, `height` | int | kích thước ảnh gửi lên |
| `regions[]` | list | mỗi vùng đã che một phần tử |
| `regions[].category` | str | một trong 12 nhóm PII |
| `regions[].polygon` | 4×2 float | toạ độ pixel trên ảnh gửi lên |
| `regions[].text` | str \| null | nội dung đọc được; `null` với `face` / `signature` |
| `masked_image` | str | JPEG mã hoá base64, cùng kích thước ảnh gốc |
| `processing_time_ms` | int | thời gian xử lý phía server |

Các nhóm được che: `person_name`, `date_of_birth`, `place_of_birth`,
`id_number`, `phone_number`, `email`, `address`, `bank_account`, `face`,
`signature`, `mrz`, `ambiguous`.

Lỗi: `400` khi tệp rỗng hoặc không giải mã được thành ảnh. Ảnh hỏng không bao
giờ làm dịch vụ sập — handler bắt và trả lỗi có kiểm soát.

---

## Cách hoạt động

```
bytes tải lên
   │
   ├─ cv2.imdecode ──────────────► 400 nếu không phải ảnh
   │
   ├─ YOLO26s-OBB (ONNX)  ───────► 13 lớp, hộp xoay; NMS nằm sẵn trong graph
   │     letterbox 640 → chạy → ánh xạ ngược về toạ độ ảnh gốc
   │
   ├─ với từng hộp có chữ:
   │     biến đổi phối cảnh hộp xoay ──► VietOCR ──► text
   │     (chỉ thử thêm 180° khi độ tin cậy < 0.5)
   │
   ├─ face / signature: dùng hộp của detector, text = null
   │
   ├─ tô đặc màu trắng mọi hộp cần che
   │
   └─ JPEG q=92 → base64 → JSON
```

### Mô hình

| Tệp | Kích thước | Vai trò |
|---|---|---|
| `models/yolo26s_obb_signature.onnx` | 38 MB | detector OBB 13 lớp, export kèm `nms=True` |
| `models/vgg_transformer.pth` | 145 MB | trọng số nhận dạng chữ VietOCR |
| `models/vietocr_vgg_transformer.yml` | 2 KB | cấu hình VietOCR gộp sẵn, để khởi động không gọi `vocr.vn` |

Graph ONNX được export từ `../signature_training/yolo26s_obb_signature.pt`:

```bash
cd ../signature_training
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('yolo26s_obb_signature.pt').export(
    format='onnx', imgsz=640, nms=True, conf=0.05, iou=0.7,
    max_det=300, opset=17, simplify=True, dynamic=False)
"
cp yolo26s_obb_signature.onnx ../ID_image_train/models/
```

Đầu ra có dạng `(1, 300, 7)` = `[cx, cy, w, h, conf, cls, angle]` trong toạ độ
letterbox. Vì NMS đã nằm trong graph, lúc chạy chỉ cần `onnxruntime` — không cần
torch, không cần ultralytics.

Trọng số VietOCR là checkpoint công khai `vgg_transformer`; cấu hình được tạo
một lần bằng `Cfg.load_config_from_name("vgg_transformer").save(...)` rồi nạp
từ đĩa bằng `Cfg.load_config_from_file`, kèm
`cfg["cnn"]["pretrained"] = False` để torchvision không tải backbone VGG19.

### Bảo đảm chạy offline

`docker run --network none` khởi động sạch và phục vụ `/health`, `/docs`,
`/static/*`, `/v1/mask`. Mọi thứ có thể gọi mạng đều được ghim cục bộ:

- detector ONNX là một tệp,
- trọng số VietOCR là một tệp (`cfg["weights"]` là đường dẫn, không phải URL),
- cấu hình VietOCR là một tệp,
- backbone pretrained của torchvision bị tắt,
- tài nguyên Swagger UI được đóng gói trong `static/`.

Kiểm chứng bằng cách xoá cache tải về và chạy không mạng:

```bash
docker run -d --name pii --gpus all --network none pii-mask:1
docker exec pii python -c "
import urllib.request
for p in ('/health','/docs','/static/swagger-ui.css','/openapi.json'):
    print(p, urllib.request.urlopen('http://127.0.0.1:8000'+p).status)
"
docker rm -f pii
```

---

## Hiệu năng

Đo trên RTX 4050 Laptop (6 GB), trong container, không mạng:

| Ảnh | Thời gian |
|---|---|
| Căn cước Estonia (9 vùng) | 282–761 ms |
| Chứng từ y tế Việt Nam (10 vùng) | 679 ms |
| Mẫu 9 ảnh nhiều nguồn | trung bình 1.37 s, cao nhất 2.35 s |

Thoải mái dưới mức trung bình 3 giây/ảnh. Request đầu tiên sau khi khởi động
chậm hơn do tạo CUDA context; các request sau mới phản ánh trạng thái ổn định.

## Cấu trúc dự án

```
ID_image_train/
├── app.py               FastAPI (routes, lifespan, kiểm tra đầu vào)
├── pipeline.py          detect → OCR → che → payload
├── run_onnx.py          wrapper detector ONNX (letterbox, giải mã OBB)
├── models/              detector + trọng số/cấu hình VietOCR
├── static/              tài nguyên Swagger UI (để /docs chạy offline)
├── utils/               hàm liệt kê ảnh
├── docs/                ghi chú thiết kế
├── Dockerfile
├── pyproject.toml / uv.lock
└── candidate/           bộ ảnh có nhãn để đánh giá (không nằm trong image)
```

## Hạn chế đã biết

- **Bố cục lạ.** Detector được huấn luyện trên 13 loại giấy tờ của bộ dữ liệu;
  giấy tờ có bố cục chưa từng thấy có thể bị bỏ sót. Chữ chỉ được đọc trong các
  hộp phát hiện được, nên bỏ sót hộp là bỏ sót trường.
- **Dấu ngoài tiếng Việt.** Từ vựng của VietOCR là tiếng Việt; các ký tự như
  `õ` (Estonia), `ä` (Phần Lan), `ñ` (Tây Ban Nha) có thể bị mất hoặc bị thay
  bằng ký tự gần giống.
- **Một worker.** Với card 6 GB chỉ đủ một bản mô hình; chạy uvicorn với đúng
  một worker.
- **Kích thước image.** Khoảng 20 GB do PyTorch kéo theo các gói CUDA.
