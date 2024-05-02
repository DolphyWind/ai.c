// Unused because they won't work with the current design. Insted, I've added dropout_probability parameter to dense layers
#include "dropout.h"
#include "layer.h"
#include "matrix.h"
#include "util.h"
#include <assert.h>

void dropout_init_layer_neurons(Layer* l, size_t batch_size, int init_weights)
{
    Dropout* d = (Dropout*)l;
    Layer* next = l->next;
    assert(next);

    Matrix* next_weights = next->get_weight_matrix(next);
    Matrix* next_biasses = next->get_bias_matrix(next);
    assert(next_biasses && next_weights);

    d->weights = matrix_init(next_weights->rows, next_weights->cols);
    d->before_activation = matrix_init(next_weights->rows, next_weights->cols);
    d->biasses = matrix_init(next_biasses->rows, next_biasses->cols);
}

Dropout* dropout_init(Layer* prev, Layer* next, cell_t dropout_probability)
{
    Dropout* d = malloc(sizeof(Dropout));

    layer_set_params(&d->base, prev, next, NULL, DROPOUT);
    d->dropout_probability = dropout_probability;
    d->base.forward_pass = dropout_forward;
    d->base.backward_pass = dropout_backward;
    d->base.init_layer_neurons = dropout_init_layer_neurons;
    d->base.print_info = dropout_print_info;
    d->base.get_neuron_matrix = dropout_get_neuron_matrix;
    d->base.get_weight_matrix = dropout_get_weight_matrix;
    d->base.get_bias_matrix = dropout_get_bias_matrix;
    d->base.get_before_activation_matrix = dropout_get_before_activation_matrix;
    d->base.free = dropout_free;

    return d;
}

void dropout_free(Layer* dropout)
{
    Dropout* d = (Dropout*)dropout;
    free(d);
}

Matrix* dropout_forward(Layer* dropout)
{
    if(dropout->is_predicting) return NULL;

    Dropout* d = (Dropout*)dropout;
    Layer* next = dropout->next;
    Matrix* next_weights = next->get_weight_matrix(next);
    Matrix* weights = dropout->get_weight_matrix(dropout);

    for(size_t y = 0; y < weights->rows; ++y)
    {
        for(size_t x = 0; x < weights->cols; ++x)
        {
            if(get_random_bounded(0, 1) < d->dropout_probability)
            {
                matrix_set(next_weights, y, x, 0);
            }
            else
            {
                matrix_set(next_weights, y, x, matrix_at(weights, y, x));
            }
        }
    }
    return NULL;
}

void dropout_backward(Layer* dropout, Matrix* gradient)
{
    return dropout->prev->backward_pass(dropout->prev, gradient);
}

size_t dropout_print_info(Layer* dropout, FILE* stream)
{
    Dropout* d = (Dropout*)dropout;
    fprintf(stream, "A dropout layer with dropout probability of %lf\n\n", d->dropout_probability);
    return 0;
}

Matrix* dropout_get_neuron_matrix(Layer* layer)
{
    return NULL;
}

Matrix* dropout_get_weight_matrix(Layer* layer)
{
    Dropout* d = (Dropout*)layer;
    return d->weights;
}

Matrix* dropout_get_bias_matrix(Layer* layer)
{
    Dropout* d = (Dropout*)layer;
    return d->biasses;
}

Matrix* dropout_get_before_activation_matrix(Layer* layer)
{
    Dropout* d = (Dropout*)layer;
    return d->before_activation;
}
