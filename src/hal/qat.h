#ifndef QAT_H
#define QAT_H

#include "quantization.h"

// QAT configuration
typedef struct {
    QuantConfig quant_config;    // Basic quantization configuration
    float learning_rate;         // Learning rate
    float smooth_factor;         // Moving average smoothing factor
    int update_step;            // Quantization parameter update step
    int calibration_steps;      // Calibration steps
    int fake_quant;             // Whether to use fake quantization
} QATConfig;

// QAT state
typedef struct {
    QuantParams* params;         // Quantization parameters array
    float* running_min;          // Running minimum value
    float* running_max;          // Running maximum value
    size_t num_tensors;          // Number of tensors
    size_t current_step;         // Current step
    int is_calibrating;          // Whether in calibration phase
} QATState;

// Initialize QAT
int qat_init(QATState** state, size_t num_tensors, const QATConfig* config);

// Clean up QAT resources
void qat_cleanup(QATState* state);

// Quantization operation in forward propagation
int qat_forward_quant(QATState* state, 
                     size_t tensor_idx,
                     float* data,
                     size_t size,
                     const QATConfig* config);

// Quantization gradient calculation in backward propagation
int qat_backward_quant(QATState* state,
                      size_t tensor_idx,
                      float* grad_output,
                      const float* grad_input,
                      const float* original_input,
                      size_t size,
                      const QATConfig* config);

// Update quantization parameters
int qat_update_params(QATState* state, const QATConfig* config);

// Get quantization parameters
const QuantParams* qat_get_params(const QATState* state, size_t tensor_idx);

// Save QAT state
int qat_save_state(const QATState* state, const char* path);

// Load QAT state
int qat_load_state(QATState* state, const char* path);

#endif // QAT_H