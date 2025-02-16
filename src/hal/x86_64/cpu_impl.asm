; x86_64 CPU implementation for matrix multiply and vector add operations

section .text

; Matrix multiplication implementation
; void matrix_multiply_asm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k)
; Parameters:
;   rdi: pointer to matrix a
;   rsi: pointer to matrix b
;   rdx: pointer to result matrix c
;   rcx: m (rows of a)
;   r8:  n (columns of b)
;   r9:  k (columns of a / rows of b)
global matrix_multiply_asm
matrix_multiply_asm:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; Save parameters
    mov     r10, rdi        ; a
    mov     r11, rsi        ; b
    mov     r12, rdx        ; c
    mov     r13, rcx        ; m
    mov     r14, r8         ; n
    mov     r15, r9         ; k

    ; Initialize outer loop counter (i)
    xor     rax, rax
.loop_i:
    cmp     rax, r13
    jge     .end_loop_i

    ; Initialize middle loop counter (j)
    xor     rbx, rbx
.loop_j:
    cmp     rbx, r14
    jge     .end_loop_j

    ; Initialize accumulator
    vxorps  ymm0, ymm0, ymm0

    ; Initialize inner loop counter (l)
    xor     rcx, rcx
.loop_l:
    cmp     rcx, r15
    jge     .end_loop_l

    ; Calculate indices
    mov     rdi, rax        ; i
    imul    rdi, r15        ; i * k
    add     rdi, rcx        ; i * k + l
    shl     rdi, 2          ; Convert to bytes
    
    mov     rsi, rcx        ; l
    imul    rsi, r14        ; l * n
    add     rsi, rbx        ; l * n + j
    shl     rsi, 2          ; Convert to bytes

    ; Load and multiply
    vbroadcastss ymm1, [r10 + rdi]    ; Broadcast a[i][l]
    vmovups ymm2, [r11 + rsi]         ; Load 8 elements from b[l][j]
    vfmadd231ps ymm0, ymm1, ymm2      ; acc += a[i][l] * b[l][j]

    inc     rcx
    jmp     .loop_l

.end_loop_l:
    ; Store result
    mov     rdi, rax        ; i
    imul    rdi, r14        ; i * n
    add     rdi, rbx        ; i * n + j
    shl     rdi, 2          ; Convert to bytes
    vmovups [r12 + rdi], ymm0

    add     rbx, 8
    jmp     .loop_j

.end_loop_j:
    inc     rax
    jmp     .loop_i

.end_loop_i:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; Vector addition implementation
; void vector_add_asm(const float* a, const float* b, float* c, size_t size)
; Parameters:
;   rdi: pointer to vector a
;   rsi: pointer to vector b
;   rdx: pointer to result vector c
;   rcx: size of vectors
global vector_add_asm
vector_add_asm:
    push    rbp
    mov     rbp, rsp

    ; Calculate number of 8-float blocks
    mov     rax, rcx
    shr     rax, 3          ; divide by 8

    ; Process 8 floats at a time using AVX
    xor     rcx, rcx
.loop:
    cmp     rcx, rax
    jge     .remainder

    vmovups ymm0, [rdi + rcx * 32]
    vmovups ymm1, [rsi + rcx * 32]
    vaddps  ymm2, ymm0, ymm1
    vmovups [rdx + rcx * 32], ymm2

    inc     rcx
    jmp     .loop

.remainder:
    ; Handle remaining elements
    mov     rcx, rax
    shl     rcx, 3          ; multiply by 8
    mov     rax, rcx

.loop_remainder:
    cmp     rax, rcx
    jge     .end

    movss   xmm0, [rdi + rax * 4]
    addss   xmm0, [rsi + rax * 4]
    movss   [rdx + rax * 4], xmm0

    inc     rax
    jmp     .loop_remainder

.end:
    pop     rbp
    ret