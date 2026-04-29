#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Definizione manuale di M_PI se non definito
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double peak_frequency;
    double peak_phase;
    double peak_period;
    int start_rebuilt_signal_index;
} DominantCycle;

// Funzione per calcolare l'errore quadratico medio
static double mean_squared_error(double *arr1, double *arr2, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = arr1[i] - arr2[i];
        sum += diff * diff;
    }
    return sum / length;
}

// Funzione per trovare massimi relativi
void find_relative_maxima(const double *data, int length, int *maxes, int *num_maxes) {
    *num_maxes = 0;
    for (int i = 1; i < length - 1; i++) {
        if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
            maxes[(*num_maxes)++] = i;
        }
    }
}

// Funzione per trovare minimi relativi
void find_relative_minima(const double *data, int length, int *mins, int *num_mins) {
    *num_mins = 0;
    for (int i = 1; i < length - 1; i++) {
        if (data[i] < data[i - 1] && data[i] < data[i + 1]) {
            mins[(*num_mins)++] = i;
        }
    }
}


void min_max_scale(double *data, int length, double output_min, double output_max) {
    double min_val = data[0];
    double max_val = data[0];
    
    // Trova i valori minimi e massimi
    for (int i = 1; i < length; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    // Esegui la normalizzazione
    double input_range = max_val - min_val;
    double output_range = output_max - output_min;

    if (input_range == 0.0) {
        for (int i = 0; i < length; i++) {
            data[i] = (output_min + output_max) / 2.0;
        }
    } else {
        for (int i = 0; i < length; i++) {
            data[i] = ((data[i] - min_val) / input_range) * output_range + output_min;
        }
    }

}



// Funzione principale per valutare la fitness
static PyObject* evaluate_fitness(PyObject *self, PyObject *args) { // PyObject *self, 
    PyObject *individual_obj, *reference_data_obj, *dominant_cycles_obj;
    int len_series, best_fit_start_back_period;
    int period_related_rebuild_range;
    double period_related_rebuild_multiplier;
    char *fitness_type;
    int return_list_type;
                                  
    if (!PyArg_ParseTuple(args,                            
                          "OOOiiidsi",  
                          &individual_obj, 
                          &reference_data_obj, 
                          &dominant_cycles_obj,
                          &len_series, 
                          &best_fit_start_back_period, 
                          &period_related_rebuild_range,
                          &period_related_rebuild_multiplier, 
                          &fitness_type, 
                          &return_list_type
                         )
       ) {
        PyErr_SetString(PyExc_TypeError, "Error parsing arguments.");
        
        printf("Error parsing arguments.\n");
        printf("Expected types: individual_obj (numpy array), reference_data_obj (numpy array), dominant_cycles_obj (list),\n");
        printf("len_series (int), best_fit_start_back_period (int), period_related_rebuild_range (int),\n");
        printf("period_related_rebuild_multiplier (double), fitness_type (char*), return_list_type (int).\n");
        fflush(stdout); 


        return NULL;
    }


    // Convert Python objects to C arrays
    PyObject *individual_array = PyArray_FROM_OTF(individual_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *reference_data_array = PyArray_FROM_OTF(reference_data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (individual_array == NULL || reference_data_array == NULL) {
        Py_XDECREF(individual_array);
        Py_XDECREF(reference_data_array);
        return NULL;
    }

    double *amplitudes = (double*)PyArray_DATA((PyArrayObject*)individual_array);
    double *reference_data = (double*)PyArray_DATA((PyArrayObject*)reference_data_array);
    

    // Process dominant cycles
    int num_cycles = PyList_Size(dominant_cycles_obj);
    DominantCycle *cycles = malloc(num_cycles * sizeof(DominantCycle));
    for (int i = 0; i < num_cycles; i++) {
        PyObject *cycle = PyList_GetItem(dominant_cycles_obj, i);
        cycles[i].peak_frequency = PyFloat_AsDouble(PyDict_GetItemString(cycle, "peak_frequencies"));
        cycles[i].peak_phase = PyFloat_AsDouble(PyDict_GetItemString(cycle, "peak_phases"));
        cycles[i].peak_period = PyFloat_AsDouble(PyDict_GetItemString(cycle, "peak_periods"));
        cycles[i].start_rebuilt_signal_index = PyLong_AsLong(PyDict_GetItemString(cycle, "start_rebuilt_signal_index"));
    }
    
    // debug
    for (int i = 0; i < PyArray_SIZE((PyArrayObject*)reference_data_array); i++) {
        if (isnan(reference_data[i])) {
            PyErr_SetString(PyExc_ValueError, "NaN detected in reference_data");
            Py_DECREF(individual_array);
            Py_DECREF(reference_data_array);
            free(cycles);
            return NULL;
        }
    }

    // Allocate memory for composite signal
    double *composite_signal = calloc(len_series, sizeof(double));

    // Rebuild composite signal
    for (int i = 0; i < num_cycles; i++) {
        int start = cycles[i].start_rebuilt_signal_index;
        int length = len_series - start;

        double amp = amplitudes[i];
        double freq = cycles[i].peak_frequency;
        double phase = cycles[i].peak_phase;

        if (isnan(amp) || isnan(freq) || isnan(phase)) {
            printf("⚠️ NaN in input to sine:\n");
            printf("  i = %d, start = %d, len_series = %d\n", i, start, len_series);
            printf("  amp = %f, freq = %f, phase = %f\n", amp, freq, phase);
            printf("  cycle period = %f\n", cycles[i].peak_period);
            fflush(stdout); 
            exit(1);
        }

        
        for (int j = start; j < len_series; j++) {
            double t = j - start;
            double value = amplitudes[i] * sin(2 * M_PI * cycles[i].peak_frequency * t + cycles[i].peak_phase);
            if (isnan(value)) {
                printf("❌ NaN in value after sin():\n");
                printf("  i = %d, t = %f, amp = %f, freq = %f, phase = %f\n", i, t, amp, freq, phase);
                fflush(stdout); 
                exit(1);
            }
            
            if (period_related_rebuild_range) {
                int period_related_rebuild_index = (int)(len_series - (cycles[i].peak_period * period_related_rebuild_multiplier));
                
//                 printf("\nperiod_related_rebuild_multiplier = %f", period_related_rebuild_multiplier);
//                 fflush(stdout); 
                
                if (period_related_rebuild_index > j) { 
//                     printf("  period_related_rebuild_index %d > j %d, len_series %d, summing 0!", period_related_rebuild_index, j, len_series);
//                     fflush(stdout); 
                    
                    composite_signal[j] += 0;
                }
                else{
                    
                    composite_signal[j] += value;
//                     printf("  period_related_rebuild_index %d >= j %d, len_series %d, summing %f!!", period_related_rebuild_index, j, len_series, value);
//                     fflush(stdout);
                }
            }
            
            else{
                
                composite_signal[j] += value;
            }
            
            
            if (isnan(composite_signal[j])) {
                printf("❌ NaN in composite_signal[%d], i = %d after sin():\n", j, i);
                printf("  i = %d, t = %f, amp = %f, freq = %f, phase = %f\n", i, t, amp, freq, phase);
                fflush(stdout); 
                exit(1);
            }
        }
        

    }

    // Calculate fitness
    int max_pos = 0;
    
    if(best_fit_start_back_period == 0){

        
        for (int i = 0; i < num_cycles; i++) {
            if (cycles[i].start_rebuilt_signal_index > max_pos) {
                max_pos = cycles[i].start_rebuilt_signal_index;
            }
        }
        
    }
    else{

        max_pos = len_series - best_fit_start_back_period; 
    }    
    
      
    // debug
    if (max_pos >= len_series) {
        PyErr_SetString(PyExc_ValueError, "max_pos is out of bounds");
        Py_DECREF(individual_array);
        Py_DECREF(reference_data_array);
        free(cycles);
        free(composite_signal);
        return NULL;
    }

    

    double fitness;

    if (strcmp(fitness_type, "mse") == 0) {
        
        // Esegui la normalizzazione sul segnale composito        
        min_max_scale(&composite_signal[max_pos], len_series - max_pos, -1, 1);


        for (int i = 0; i < len_series - max_pos; i++) {
            composite_signal[max_pos + i] *= 100;  // Moltiplica per 100 come nel codice Python
        }
        
        // debug
        for (int i = max_pos; i < len_series; i++) {
            if (isnan(composite_signal[i])) {
                char error_msg[128];
                sprintf(error_msg, "NaN detected in composite_signal at i = %d", i);
                PyErr_SetString(PyExc_ValueError, error_msg);
                Py_DECREF(individual_array);
                Py_DECREF(reference_data_array);
                free(cycles);
                free(composite_signal);
                return NULL;
            }
        }


        fitness = mean_squared_error(&reference_data[max_pos], &composite_signal[max_pos], len_series - max_pos);

    } else if (strcmp(fitness_type, "just_mins_maxes") == 0) {
        


        int *mins = malloc(len_series * sizeof(int));
        int *maxes = malloc(len_series * sizeof(int));
        int mins_length = 0, maxes_length = 0;

        find_relative_maxima(&composite_signal[max_pos], len_series - max_pos, maxes, &maxes_length);
        find_relative_minima(&composite_signal[max_pos], len_series - max_pos, mins, &mins_length);

        int total_peaks = mins_length + maxes_length;
        int *peaks_indexes = malloc(total_peaks * sizeof(int));
        memcpy(peaks_indexes, mins, mins_length * sizeof(int));
        memcpy(peaks_indexes + mins_length, maxes, maxes_length * sizeof(int));

        double *scaled_composite_peaks = malloc(total_peaks * sizeof(double));
        double *reference_data_peaks = malloc(total_peaks * sizeof(double));

        for (int i = 0; i < total_peaks; i++) {
            scaled_composite_peaks[i] = composite_signal[max_pos + peaks_indexes[i]];
            reference_data_peaks[i] = reference_data[max_pos + peaks_indexes[i]];
        }

        // Esegui la normalizzazione sui picchi
        min_max_scale(scaled_composite_peaks, total_peaks, -1, 1);

        for (int i = 0; i < total_peaks; i++) {
            scaled_composite_peaks[i] *= 100;  // Moltiplica per 100 come nel codice Python
        }

        fitness = mean_squared_error(reference_data_peaks, scaled_composite_peaks, total_peaks);

        free(mins);
        free(maxes);
        free(peaks_indexes);
        free(scaled_composite_peaks);
        free(reference_data_peaks);
    }

    free(composite_signal);
    free(cycles);
    Py_DECREF(individual_array);
    Py_DECREF(reference_data_array);

    if (return_list_type) {
        PyObject *result_list = PyTuple_New(1);
        PyTuple_SetItem(result_list, 0, Py_BuildValue("d", fitness));
        return result_list;
    } else {
        return Py_BuildValue("d", fitness);
    }
}

// Definizione dei metodi del modulo
static PyMethodDef GeneticOptimizationMethods[] = {
    {"evaluate_fitness", evaluate_fitness, METH_VARARGS, "Evaluate fitness."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Definizione del modulo
static struct PyModuleDef geneticoptimizationmodule = {
    PyModuleDef_HEAD_INIT,
    "genetic_optimization",  // Nome del modulo
    NULL,  // Documentazione del modulo
    -1,  // Stato del modulo globale
    GeneticOptimizationMethods
};

// Funzione di inizializzazione del modulo
PyMODINIT_FUNC PyInit_genetic_optimization(void) {
    PyObject* module;

    module = PyModule_Create(&geneticoptimizationmodule);
    if (module == NULL) {
        return NULL;
    }

    import_array();  // Necessario per l'integrazione con NumPy
    return module;
}
