; x86_64 训练相关汇编实现

section .text
global backward_matrix_multiply_asm
global backward_vector_add_asm
global compute_loss_asm
global backward_loss_asm
global optimizer_step_sgd_asm
global optimizer_step_adam_asm

; void backward_matrix_multiply_asm(const float* grad_output, const float* input,
;                                 float* grad_input, float* grad_weight,
;                                 size_t m, size_t n, size_t k)
backward_matrix_multiply_asm:
    push rbp
    mov rbp, rsp
    
    ; 保存必要的寄存器
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; 参数映射:
    ; rdi = grad_output
    ; rsi = input
    ; rdx = grad_input
    ; rcx = grad_weight
    ; r8 = m
    ; r9 = n
    ; [rbp+16] = k
    
    ; 使用AVX指令计算梯度
    xor r10, r10        ; i = 0
    
.loop_i:
    xor r11, r11        ; j = 0
    
.loop_j:
    vxorps ymm0, ymm0, ymm0   ; 清零累加寄存器
    
    mov r12, 0          ; k = 0
.loop_k:
    ; 计算grad_input
    vmovups ymm1, [rdi + r10*4]    ; 加载grad_output
    vmovups ymm2, [rsi + r11*4]    ; 加载input
    vfmadd231ps ymm0, ymm1, ymm2   ; ymm0 += grad_output * input
    
    add r12, 8
    cmp r12, [rbp+16]
    jl .loop_k
    
    ; 存储grad_weight
    vmovups [rcx + r11*4], ymm0
    
    add r11, 8
    cmp r11, r9
    jl .loop_j
    
    add r10, 1
    cmp r10, r8
    jl .loop_i
    
    ; 恢复寄存器
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    
    mov rsp, rbp
    pop rbp
    ret

; void optimizer_step_adam_asm(float* params, float* grads, float* m, float* v,
;                            float beta1, float beta2, float lr, float epsilon,
;                            size_t size)
optimizer_step_adam_asm:
    push rbp
    mov rbp, rsp
    
    ; 参数映射:
    ; rdi = params
    ; rsi = grads
    ; rdx = m (一阶矩)
    ; rcx = v (二阶矩)
    ; xmm0 = beta1
    ; xmm1 = beta2
    ; xmm2 = lr
    ; xmm3 = epsilon
    ; r8 = size
    
    ; 广播beta1和beta2到向量寄存器
    vbroadcastss ymm4, xmm0  ; beta1
    vbroadcastss ymm5, xmm1  ; beta2
    vbroadcastss ymm6, xmm2  ; lr
    vbroadcastss ymm7, xmm3  ; epsilon
    
    xor rax, rax            ; i = 0
    
.loop:
    ; 加载数据
    vmovups ymm0, [rsi + rax]  ; 加载梯度
    vmovups ymm1, [rdx + rax]  ; 加载一阶矩
    vmovups ymm2, [rcx + rax]  ; 加载二阶矩
    
    ; 更新一阶矩: m = beta1 * m + (1 - beta1) * grad
    vsubps ymm8, [rel one], ymm4  ; 1 - beta1
    vmulps ymm9, ymm0, ymm8      ; (1 - beta1) * grad
    vmulps ymm1, ymm1, ymm4      ; beta1 * m
    vaddps ymm1, ymm1, ymm9      ; m = beta1 * m + (1 - beta1) * grad
    
    ; 更新二阶矩: v = beta2 * v + (1 - beta2) * grad * grad
    vmulps ymm10, ymm0, ymm0     ; grad * grad
    vsubps ymm8, [rel one], ymm5  ; 1 - beta2
    vmulps ymm9, ymm10, ymm8     ; (1 - beta2) * grad * grad
    vmulps ymm2, ymm2, ymm5      ; beta2 * v
    vaddps ymm2, ymm2, ymm9      ; v = beta2 * v + (1 - beta2) * grad * grad
    
    ; 计算更新步长: -lr * m / (sqrt(v) + epsilon)
    vsqrtps ymm3, ymm2           ; sqrt(v)
    vaddps ymm3, ymm3, ymm7      ; sqrt(v) + epsilon
    vdivps ymm3, ymm1, ymm3      ; m / (sqrt(v) + epsilon)
    vmulps ymm3, ymm3, ymm6      ; -lr * m / (sqrt(v) + epsilon)
    
    ; 更新参数
    vmovups ymm8, [rdi + rax]    ; 加载参数
    vsubps ymm8, ymm8, ymm3      ; params -= update
    vmovups [rdi + rax], ymm8    ; 存储更新后的参数
    
    ; 存储动量
    vmovups [rdx + rax], ymm1    ; 存储一阶矩
    vmovups [rcx + rax], ymm2    ; 存储二阶矩
    
    add rax, 32                  ; 更新索引 (8 * sizeof(float))
    cmp rax, r8
    jl .loop
    
    mov rsp, rbp
    pop rbp
    ret

section .data
align 32
one: dd 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 