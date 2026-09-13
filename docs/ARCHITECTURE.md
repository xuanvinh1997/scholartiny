# Chi tiết kiến trúc và các bất biến

## Mamba-2 subset

Reference triển khai recurrence rời rạc theo head, với B/C chung một nhóm:

```math
\Delta_{t,h}=\operatorname{softplus}(d_{t,h}+b_{\Delta,h}),\quad A_h=-\exp(a_h)
S_{t,h,p,n}=\exp(\Delta_{t,h}A_h)S_{t-1,h,p,n}
             +\Delta_{t,h}X_{t,h,p}B_{t,n}
Y_{t,h,p}=\sum_n S_{t,h,p,n}C_{t,n}+D_hX_{t,h,p}.
```

X/B/C đến từ depthwise causal convolution rồi SiLU. Nhánh gate z từ in_proj.
Cấu hình `norm_before_gate=False` nghĩa là RMSNorm áp dụng sau phép nhân SiLU(z).
A_log, dt_bias, D không bị weight decay. A_log và recurrence tham chiếu được tính FP32.

Đây là một cấu hình cụ thể tương ứng interface Mamba2, không bao phủ grouped B/C,
partial d_ssm, tensor parallel, variable-length packed inference hay mọi tùy chọn upstream.
Reference dùng vòng lặp theo token và autograd thông thường; upstream dùng kernel SSD.
Vì vậy parity toán học và parity số học cần được kiểm tra trên backend GPU, không suy ra
chỉ từ CPU self-consistency tests. Test GPU đã có nhưng bị skip trong môi trường giao gói.

## Loss

Labels luôn **chưa shift**, cùng shape input_ids. Forward so sánh `logits[:, :-1]` với
`labels[:, 1:]`. Mask -100 che system/user/tool/padding. Loss được cộng theo token rồi
chia số token được giám sát, không chia đơn giản theo số example hoặc số microbatch.

Trong DDP, mỗi microbatch backward dùng `loss_sum × world_size / global_target_count`.
DDP trung bình gradient qua ranks, do đó gradient cuối bằng loss trung bình toàn cục.
Không chia thêm cho grad_accum lần nữa. Count được tính cho toàn accumulation window.

Vocab projection chia chunk; checkpoint từng projection/loss khi training để không
phải giữ toàn bộ B×T×V logits cho backward. Đây là trade-off compute/memory,
không thay thế fused cross-entropy chuyên dụng.

## Cache

Attention lưu K/V theo số KV head thật, không theo số query head. Khi gọi SDPA, hiện
implementation mở rộng K/V tạm thời bằng repeat_interleave. Đây không phải bảo đảm
backend GQA kernel tối ưu. Prefill dùng causal mask. Decode một token dùng
`is_causal=False` vì query đó cần nhìn toàn bộ cache; dùng `is_causal=True` trên ma trận
1×N có thể chỉ cho nhìn vị trí đầu theo alignment mặc định.

RoPE dùng offset toàn prefix; cache chỉ nhận dữ liệu inference, cùng batch size.
Không hỗ trợ left padding hoặc batch decode có độ dài khác nhau. Một prefix mỗi cache.
Mamba upstream cached decode chỉ nhận 1 token sau prefill; model kiểm tra điều kiện đó.
Không có sliding eviction: vượt max_seq_len thì báo lỗi, không drop token ngầm.

Hybrid memory không O(1): chỉ state của SSM có kích thước cố định; 6 attention layer
vẫn giữ KV tuyến tính theo độ dài. Training activation khác hoàn toàn inference cache.

## Causal data policy

SFT: một cuộc hội thoại mỗi hàng; padding bên phải; token thật không nhìn thấy padding
trong tương lai. Padding target bị mask và không tái sử dụng state sau batch đó.

Pretrain: văn bản nối EOS, các block có thể chứa nhiều tài liệu; EOS không tách attention
hay reset SSM. Không gọi đó là packed document isolation. Một token overlap giữa các
block tránh mất một target ở ranh giới stride; state vẫn reset mỗi block/batch.

## Tương thích

Checkpoint riêng của ScholarTiny, weights_only load. `ssm_backend` có thể đổi giữa
reference/upstream trong subset đã định nghĩa với strict state_dict; trước khi dựa vào
điều đó cho công việc thật, chạy GPU parity test. Không dùng checkpoint từ namespace
MiniMind/Qwen/Mamba khác mà không có converter được kiểm thử.
