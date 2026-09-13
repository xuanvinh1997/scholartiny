# Báo cáo kiểm thử ScholarTiny

Ngày lập báo cáo: **10-09-2026**. Kết quả dưới đây được chạy thực tế trong môi trường tạo gói, không phải chỉ là kế hoạch kiểm thử.

## 1. Môi trường

Python **3.13.5**, PyTorch **2.10.0+cpu**, NumPy 2.3.5. **Không có CUDA/GPU**. Xem `environment.json` để biết phiên bản các thư viện còn lại. Môi trường không có `tokenizers`, `datasets`, `mamba-ssm` hoặc `causal-conv1d`; cài thêm hai thư viện dữ liệu không thành công do môi trường container không phân giải được mạng.

Các kiểm thử model dùng backend `reference`. Không suy diễn kết quả CPU thành kết quả kernel CUDA chính thức.

## 2. Unit/integration tests

Lệnh đã chạy từ thư mục gốc repo:

```bash
PYTHONPATH=. python -m pytest -q --junitxml=reports/pytest.xml
```

**Kết quả lần cuối: 38 passed, 3 skipped, 10.45 giây.** Log nguyên văn: `pytest.txt`. Báo cáo máy đọc được: `pytest.xml`.

Các nhóm đã qua gồm causal masking; prefill/decode cache; chunk decode của pure attention; right-padding invariance; causal shift và assistant-only loss; gradient checkpointing; tham số SSM cần giữ initialization/no-weight-decay; tied embedding; chuẩn hóa Unicode; cấu trúc tool call; nhóm các lời giải cùng một bài toán; exact dedup/provenance; kiểm tra license metadata; lọc n-gram; collector tiếp tục từ vị trí đã lưu bằng **mock HF streaming**; mmap/batch sampler; loại SFT quá dài; công cụ toán có giới hạn; RAG và citation ID; dữ liệu tổng hợp có kết quả thực thi; RAG warmup kế thừa split/license; tiếp tục training với kết quả trọng số khớp bitwise trên CPU.

Ba test bị bỏ qua, không phải đã qua:

| Test | Lý do |
|---|---|
| BPE Unicode round trip/training | Thiếu `tokenizers` |
| Parity Mamba-2 reference với upstream | Không có CUDA |
| Backward BF16 và cache của hybrid dùng upstream | Không có CUDA |

Có thể chạy hai GPU test trên máy đích bằng `python -m pytest -q -m gpu` sau khi cài PyTorch CUDA, `mamba-ssm` và `causal-conv1d` tương thích. Không bắt đầu job lớn trước khi kiểm tra được backend này.

## 3. Smoke test toàn pipeline offline

```bash
PYTHONPATH=. python -m scripts.smoke_test --out .work/smoke
```

**Đã qua.** Log: `smoke_output.txt`; kết quả có cấu trúc: `smoke_report.json`.

Đường đi thực sự được thực thi: tạo dữ liệu gốc tự soạn → byte tokenizer debug 264 token → tokenize/mmap → **hai bước pretrain** → nạp weights sang SFT → **hai bước SFT** → sinh bốn token → tạo chỉ mục SQLite FTS5 → truy xuất → thực thi phép tính `1/3 + 1/6` trong tiến trình riêng, trả kết quả chính xác `1/2`.

Dữ liệu fixture: 80 đoạn pretrain, 16.541 token; 12 hội thoại SFT, 6.838 token và 767 vị trí được giám sát trong toàn bộ tập. Chỉ mục có 10 tài liệu/10 chunk. Bộ kiểm thử đã truy xuất hai chunk.

Nội dung model sinh ra sau vài bước này không có ý nghĩa học thuật. **Đây không phải phép đo chất lượng và không phải model pretrained để sử dụng.** Các checkpoint debug tạm trong `.work/` không được đưa vào archive.

## 4. DDP hai tiến trình CPU

Đã chạy lệnh sau, sử dụng dữ liệu debug do smoke test tạo ra:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  -m trainer.train_pretrain \
  --config configs/debug.yaml \
  --tokenizer .work/smoke/tokenizer.json \
  --data .work/smoke/packed_pretrain \
  --out .work/ddp_cpu \
  --seq-len 32 --batch-size 1 --grad-accum 2 \
  --max-steps 2 --warmup-steps 1 --save-every 2 \
  --device cpu --cpu-threads 1
```

**Đã qua trên Gloo**, hai rank kết nối, loss hữu hạn, 124 target token trên mỗi global step, 248 target token qua hai bước, checkpoint cuối được ghi. Log: `ddp_cpu.txt`. Điều này không kiểm chứng NCCL hoặc DDP trên nhiều GPU.

## 5. Đếm tham số và ước lượng cache

Đã đếm mô hình trên meta device, không cấp phát hàng trăm triệu weights thật. Kết quả đầy đủ: `model_parameters.json`.

| Config | Tham số |
|---|---:|
| debug | 209.048 |
| hybrid_59m | 59.291.760 |
| hybrid_373m | 373.381.824 |
| transformer_373m | 372.559.872 |
| hybrid_355m | 354.507.456 |
| transformer_355m | 353.685.504 |

Ở batch 1 và cache 2 byte/phần tử, context 8.192 token: **KV riêng** của hybrid 373M là 48 MiB; pure Transformer đối chứng là 192 MiB. Hybrid còn khoảng 4,80 MiB cho SSM/convolution state theo giả định cùng 2 byte. Backend reference hiện giữ SSM state bằng FP32, nên phần đó không dùng giả định 2 byte khi chạy CPU.

Đây là tính toán kích thước tensor, **không phải phép đo VRAM tổng**. Không gồm weights, activations, gradient, optimizer state, transient GQA repeat, workspace hoặc allocator. Hybrid vẫn có KV cache tăng theo độ dài context. Các con số 16K trong JSON là ngoại suy công thức, không phải chứng nhận chất lượng hoặc hỗ trợ context đã huấn luyện 16K.

## 6. Chưa được kiểm chứng hoặc chưa triển khai

Chưa tải public dataset qua mạng; chưa chạy huấn luyện BPE; chưa chạy CUDA/Triton Mamba-2; chưa có benchmark GPU throughput/VRAM; chưa pretrain model 59M/355M/373M trên corpus thật; chưa có kết quả MATH, academic reasoning, RAG faithfulness hoặc agent end-to-end.

Collector hiện được kiểm thử bằng adapter fixtures và mocked streaming; pin SHA, đọc card, kiểm tra license và truy cập nguồn gated phải được kiểm chứng trên mạng/máy đích. Public dataset không được nhúng trong archive.

`evaluate.py` đo dự đoán call plan và câu trả lời **với lịch sử/quan sát gold** từ tập kiểm tra. Citation-ID matching không chứng minh nội dung được nguồn hỗ trợ. `agent.runner` có vòng thực thi tool thật, nhưng repo chưa có bộ benchmark end-to-end cho vòng đó. Không báo các metric này như thể đã đo semantic faithfulness.

RAG warmup tự sinh là bài chọn đoạn/trích nguyên văn; không thay thế bộ academic QA có kiểm định. Chưa có semantic dedup toàn corpus, lọc PII toàn diện, DPO/GRPO, distillation teacher, phân phối checkpoint HF, export GGUF hoặc tích hợp vLLM.

Tool runtime là allowlist toán/truy xuất với AST giới hạn và timeout, không có Python tùy ý. Tiến trình con không phải ranh giới bảo mật hoàn chỉnh cho dịch vụ nhiều người thuê không tin cậy; cần thêm container isolation, giới hạn bộ nhớ và kiểm soát truy cập khi triển khai.

## 7. Kiểm tra mã nguồn

`python -m compileall -q model dataset trainer scripts agent tests` đã hoàn tất không báo lỗi. Archive chỉ chứa mã nguồn, config, tài liệu, ví dụ tự soạn và báo cáo kiểm thử; không chứa token truy cập, weights huấn luyện thật hay dữ liệu công khai được phân phối lại.

Cả 13 CLI entrypoint đã được import và gọi `--help` thành công; xem `cli_checks.json`.
