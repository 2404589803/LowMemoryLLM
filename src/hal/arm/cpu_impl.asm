.arch armv8-a
.text
.align 2

// ARM NEON 向量指令实现

.global matrix_multiply_asm
.global vector_add_asm

// void matrix_multiply_asm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k)
matrix_multiply_asm:
    // 保存寄存器
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    // 保存NEON寄存器
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    
    // 初始化循环计数器
    mov x6, xzr         // i = 0
    
.loop_i:
    mov x7, xzr         // j = 0
    
.loop_j:
    mov x8, xzr         // k = 0
    movi v0.4s, #0      // 清零累加寄存器
    movi v1.4s, #0
    
.loop_k:
    // 计算数组偏移
    mul x9, x6, x5      // i * k
    add x9, x9, x8      // i * k + k
    lsl x9, x9, #2      // (i * k + k) * 4
    
    mul x10, x8, x4     // k * n
    add x10, x10, x7    // k * n + j
    lsl x10, x10, #2    // (k * n + j) * 4
    
    // 加载数据
    ld1 {v2.4s}, [x0, x9]      // 加载a的4个float
    ld1r {v3.4s}, [x1, x10]    // 广播b的一个元素
    
    // 计算
    fmla v0.4s, v2.4s, v3.4s   // v0 += v2 * v3
    
    add x8, x8, #4             // k += 4
    cmp x8, x5                 // k < K?
    b.lt .loop_k
    
    // 存储结果
    mul x9, x6, x4             // i * n
    add x9, x9, x7             // i * n + j
    lsl x9, x9, #2             // (i * n + j) * 4
    
    st1 {v0.4s}, [x2, x9]      // 存储结果
    
    add x7, x7, #4             // j += 4
    cmp x7, x4                 // j < N?
    b.lt .loop_j
    
    add x6, x6, #1             // i++
    cmp x6, x3                 // i < M?
    b.lt .loop_i
    
    // 恢复NEON寄存器
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    
    // 恢复寄存器
    ldp x29, x30, [sp], #16
    ret

// void vector_add_asm(const float* a, const float* b, float* c, size_t size)
vector_add_asm:
    // 保存寄存器
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    // 保存NEON寄存器
    stp d8, d9, [sp, #-16]!
    
    // 计算向量块数（每块4个float）
    lsr x4, x3, #2        // size / 4
    
    // 主循环 - 每次处理4个float
.loop:
    cbz x4, .remainder
    
    // 加载数据
    ld1 {v0.4s}, [x0], #16    // 加载a的4个float
    ld1 {v1.4s}, [x1], #16    // 加载b的4个float
    
    // 执行向量加法
    fadd v2.4s, v0.4s, v1.4s
    
    // 存储结果
    st1 {v2.4s}, [x2], #16
    
    // 更新计数器
    sub x4, x4, #1
    cbnz x4, .loop
    
.remainder:
    // 处理剩余的元素
    and x4, x3, #3        // size % 4
    cbz x4, .done
    
.remainder_loop:
    // 单个元素加法
    ldr s0, [x0], #4
    ldr s1, [x1], #4
    fadd s2, s0, s1
    str s2, [x2], #4
    
    // 更新计数器
    sub x4, x4, #1
    cbnz x4, .remainder_loop
    
.done:
    // 恢复NEON寄存器
    ldp d8, d9, [sp], #16
    
    // 恢复寄存器
    ldp x29, x30, [sp], #16
    ret 