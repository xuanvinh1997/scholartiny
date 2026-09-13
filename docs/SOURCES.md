# Nguồn đối chiếu

Đối chiếu trực tuyến khi soạn gói ngày 10-09-2026. Collector sẽ resolve dataset commit
thực tế ở thời điểm chạy và lưu snapshot vào manifest. Không gán nhãn “đã tải và train”
cho những dataset chỉ mới có adapter.

## Kiến trúc và API

- MiniMind: https://github.com/jingyaogong/minimind — tham khảo cách tổ chức model,
  dataset, trainer và CLI; không vendoring code hay trọng số MiniMind.
- Mamba: https://github.com/state-spaces/mamba
- Mamba2 module: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
  — interface và tập cấu hình hỗ trợ cho adapter/reference.
- Gated normalization: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py
  — phân biệt normalize-before-gate và gate-before-normalize.
- Mamba-2/SSD paper: https://arxiv.org/abs/2405.21060
- Jamba: https://arxiv.org/abs/2403.19887 — động cơ hybrid, không phải bằng chứng rằng 3:1 tối ưu cho 350M.
- Hugging Face streaming: https://huggingface.co/docs/datasets/stream
- Tokenizers trainer API: https://huggingface.co/docs/tokenizers/api/trainers

## Dataset cards

- SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
  Card label `odc-by`. Registry dùng FineWeb-Edu-Dedup và Cosmopedia-v2. Quyền của
  underlying sources cần được xem riêng. Python-Edu có workflow khác, không giả cột text.
- Vietnamese Wikipedia snapshot: https://huggingface.co/datasets/wikimedia/wikipedia
  Dùng `20231101.vi`, card có `cc-by-sa-3.0` và `gfdl`; giữ attribution và điều kiện tương ứng.
- OpenMathInstruct-2: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
  Card label `cc-by-4.0`. Dùng `train_1M`; grouped by problem. Majority-vote augmented
  answers không phải kết quả kiểm chứng hình thức.
- xLAM/APIGen: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
  Card label `cc-by-4.0`, gated acceptance và yêu cầu citation APIGen; có thêm research-purpose
  language trong phần ethical considerations. Cần đọc đầy đủ điều khoản.
- UltraChat: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
  Card label `mit`; dùng `train_sft`, không lấy `test_sft` cho training.

Original architecture brief do người dùng cung cấp được giữ trong
`docs/INPUT_ARCHITECTURE.md`. Chi tiết như RoPE theta, convolution bias, tokenizer,
loader và tool runtime là quyết định triển khai mới, không phải kết quả benchmark
có sẵn trong bản thiết kế đó.
