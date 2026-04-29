#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Struttura invariata
typedef struct {
    double amplitude;
    double frequency;
    double phase;
    double peak_period;
    int start_index;
} DominantCycle;

static double mean_squared_error(const double* arr1, const double* arr2, int length){
    double sum = 0.0;
    for(int i=0; i<length; i++){
        double diff = arr1[i] - arr2[i];
        sum += diff * diff;
    }
    return sum / length;
}

/*
  Firma invariata: "OOOppiiidsi"
  
  1) individual_obj (PyObject*) -> array con ampiezze (e freq/fasi, se abilitati)
  2) reference_data_obj (PyObject*) -> array NumPy (double) con i dati di riferimento
  3) dominant_cycles_obj (PyObject*) -> lista di dict con, ad es., "start_rebuilt_signal_index" e "peak_periods"
  4) frequencies_ft (int)          -> se 1, in 'individual' ci sono anche le frequenze
  5) phases_ft (int)               -> se 1, in 'individual' ci sono anche le fasi
  6) len_series (int)              -> lunghezza di reference_data
  7) best_fit_start_back_period (int)
  8) period_related_rebuild_range (int)
  9) period_related_rebuild_multiplier (double)
 10) fitness_type (const char*)    -> "mse", etc.
 11) return_list_type (int)        -> se 1, restituisce (fitness,) altrimenti fitness singolo
*/

static PyObject* evaluate_fitness(PyObject* self, PyObject* args)
{
    PyObject* individual_obj = NULL;
    PyObject* reference_data_obj = NULL;
    PyObject* dominant_cycles_obj = NULL;
    int frequencies_ft = 0;
    int phases_ft = 0;
    int len_series = 0;
    int best_fit_start_back_period = 0;
    int period_related_rebuild_range = 0;
    double period_related_rebuild_multiplier = 1.0;
    const char* fitness_type = NULL;
    int return_list_type = 0;

    if(!PyArg_ParseTuple(
        args,
        "OOOppiiidsi", // la stessa firma di prima
        &individual_obj,
        &reference_data_obj,
        &dominant_cycles_obj,
        &frequencies_ft,
        &phases_ft,
        &len_series,
        &best_fit_start_back_period,
        &period_related_rebuild_range,
        &period_related_rebuild_multiplier,
        &fitness_type,
        &return_list_type
    )){
        PyErr_SetString(PyExc_TypeError, "evaluate_fitness: error parsing args.");
        return NULL;
    }

    // Converte individual_obj in array NumPy double
    PyArrayObject* individual_array = (PyArrayObject*) PyArray_FROM_OTF(individual_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!individual_array){
        PyErr_SetString(PyExc_ValueError, "Cannot convert individual_obj to NumPy float64 array.");
        return NULL;
    }
    double* individual = (double*) PyArray_DATA(individual_array);
    npy_intp individual_size = PyArray_SIZE(individual_array);

    // Converte reference_data_obj in array NumPy double
    PyArrayObject* ref_array = (PyArrayObject*) PyArray_FROM_OTF(reference_data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!ref_array){
        Py_XDECREF(individual_array);
        PyErr_SetString(PyExc_ValueError, "Cannot convert reference_data_obj to NumPy float64 array.");
        return NULL;
    }
    double* reference_data = (double*) PyArray_DATA(ref_array);

    // Controlla che dominant_cycles_obj sia una lista
    if(!PyList_Check(dominant_cycles_obj)){
        PyErr_SetString(PyExc_TypeError, "dominant_cycles_obj must be a list of dicts.");
        Py_DECREF(individual_array);
        Py_DECREF(ref_array);
        return NULL;
    }

    int n = (int)PyList_Size(dominant_cycles_obj);
    if(n <= 0){
        // se non ci sono cicli, restituiamo fitness grande
        Py_DECREF(individual_array);
        Py_DECREF(ref_array);
        return Py_BuildValue("d", 1e9);
    }

    // Calcoliamo quanto deve essere lungo 'individual'
    // Se frequencies_ft==1, abbiamo n valori in più
    // Se phases_ft==1, abbiamo altri n valori in più
    // Quindi base n (ampiezze), +n freq, +n fasi
    int expected_size = n; // ampiezze
    if(frequencies_ft) expected_size += n; 
    if(phases_ft)      expected_size += n;

    if((int)individual_size != expected_size){
        PyErr_Format(PyExc_ValueError,
            "evaluate_fitness: individual size mismatch. Expected %d, got %d.",
            expected_size, (int)individual_size);
        Py_DECREF(individual_array);
        Py_DECREF(ref_array);
        return NULL;
    }

    // Allochiamo i cicli
    DominantCycle* cycles = (DominantCycle*) malloc(n * sizeof(DominantCycle));
    if(!cycles){
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate cycles array.");
        Py_DECREF(individual_array);
        Py_DECREF(ref_array);
        return NULL;
    }

    // Copiamo da 'individual' le ampiezze, freq, fasi
    int offset = 0;
    // Ampiezze
    for(int i=0; i<n; i++){
        cycles[i].amplitude = individual[offset++];
    }

    // Frequenze, se abilitate
    if(frequencies_ft){
        for(int i=0; i<n; i++){
            cycles[i].frequency = individual[offset++];
        }
    } else {
        // Se non stai ottimizzando le frequenze, potresti:
        // - prenderle dal dict (peak_frequencies)
        // - impostare un valore fisso
        // - etc.
        for(int i=0; i<n; i++){
            PyObject* cycleDict = PyList_GetItem(dominant_cycles_obj, i);
            PyObject* freqObj = PyDict_GetItemString(cycleDict, "peak_frequencies");
            if(!freqObj || !PyFloat_Check(freqObj)){
                // fallback, errore, ...
                cycles[i].frequency = 0.0; // o come preferisci
            } else {
                cycles[i].frequency = PyFloat_AsDouble(freqObj);
            }
        }
    }

    // Fasi, se abilitate
    if(phases_ft){
        for(int i=0; i<n; i++){
            cycles[i].phase = individual[offset++];
        }
    } else {
        // fallback sul dict (peak_phases) o 0.0
        for(int i=0; i<n; i++){
            PyObject* cycleDict = PyList_GetItem(dominant_cycles_obj, i);
            PyObject* phObj = PyDict_GetItemString(cycleDict, "peak_phases");
            if(!phObj || !PyFloat_Check(phObj)){
                cycles[i].phase = 0.0;
            } else {
                cycles[i].phase = PyFloat_AsDouble(phObj);
            }
        }
    }

    // Ora leggiamo da python i campi minimi: start_rebuilt_signal_index, peak_periods
    for(int i=0; i<n; i++){
        PyObject* cycleDict = PyList_GetItem(dominant_cycles_obj, i);
        // start_index
        PyObject* sIdx = PyDict_GetItemString(cycleDict, "start_rebuilt_signal_index");
        if(!sIdx || !PyLong_Check(sIdx)){
            // fallback o errore
            cycles[i].start_index = 0; 
        } else {
            cycles[i].start_index = (int)PyLong_AsLong(sIdx);
        }

        // peak_period
        PyObject* pp = PyDict_GetItemString(cycleDict, "peak_periods");
        if(!pp || !PyFloat_Check(pp)){
            // se manca, calcoliamo 1.0/freq
            if(cycles[i].frequency != 0.0)
                cycles[i].peak_period = 1.0 / cycles[i].frequency;
            else
                cycles[i].peak_period = 1e6; 
        } else {
            cycles[i].peak_period = PyFloat_AsDouble(pp);
        }
    }

    // Creiamo il segnale composito
    double* composite = (double*)calloc(len_series, sizeof(double));
    if(!composite){
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate composite array.");
        free(cycles);
        Py_DECREF(individual_array);
        Py_DECREF(ref_array);
        return NULL;
    }

    // Fill composite
    for(int i=0; i<n; i++){
        int start = cycles[i].start_index;
        if(start >= len_series) continue;

        for(int j=start; j<len_series; j++){
            double t = (double)(j - start);
            double val = cycles[i].amplitude * sin(2.0*M_PI*cycles[i].frequency * t + cycles[i].phase);

            if(period_related_rebuild_range){
                int cutoff = (int)(len_series - cycles[i].peak_period * period_related_rebuild_multiplier);
                if(cutoff < 0) cutoff = 0;
                if(j >= cutoff){
                    composite[j] += val;
                }
            } else {
                composite[j] += val;
            }
        }
    }

    // Determina max_pos
    int max_pos = 0;
    if(best_fit_start_back_period == 0){
        // prendi lo start_index max
        for(int i=0; i<n; i++){
            if(cycles[i].start_index > max_pos) max_pos = cycles[i].start_index;
        }
    } else {
        int alt_pos = len_series - best_fit_start_back_period;
        if(alt_pos > max_pos) max_pos = alt_pos;
    }
    if(max_pos >= len_series) max_pos = 0; // fallback
    int compare_len = len_series - max_pos;
    if(compare_len <= 0){
        free(composite);
        free(cycles);
        Py_DECREF(individual_array);
        Py_DECREF(ref_array);
        return Py_BuildValue("d", 1e9);
    }

/*
    // Calcolo fitness, es. MSE
    double fitness_val = 0.0;
    if(strcmp(fitness_type, "mse")==0){
        fitness_val = mean_squared_error(&reference_data[max_pos], &composite[max_pos], compare_len);
    } else {
        // fallback su MSE
        fitness_val = mean_squared_error(&reference_data[max_pos], &composite[max_pos], compare_len);
    }
*/

	// --- Normalize composite[max_pos:] in range [-1, 1] and multiply by 100 ---
	double c_min = composite[max_pos];
	double c_max = composite[max_pos];
	for (int i = max_pos + 1; i < len_series; i++) {
		if (composite[i] < c_min) c_min = composite[i];
		if (composite[i] > c_max) c_max = composite[i];
	}
	double c_range = c_max - c_min;
	if (c_range > 0.0) {
		for (int i = max_pos; i < len_series; i++) {
			composite[i] = ((composite[i] - c_min) / c_range) * 2.0 - 1.0;
		}
	}
	for (int i = max_pos; i < len_series; i++) {
		composite[i] *= 100.0;
	}

	// Calcolo fitness con segnale normalizzato
	double fitness_val = mean_squared_error(&reference_data[max_pos], &composite[max_pos], compare_len);


    free(composite);
    free(cycles);
    Py_DECREF(individual_array);
    Py_DECREF(ref_array);

    // Ritorno
    if(return_list_type){
        // stile DEAP => (fitness,)
        PyObject* result = PyTuple_New(1);
        PyTuple_SetItem(result, 0, Py_BuildValue("d", fitness_val));
        return result;
    } else {
        // singolo double
        return Py_BuildValue("d", fitness_val);
    }
}

// Tabella metodi
static PyMethodDef CyFitnessMethods[] = {
    {"evaluate_fitness", evaluate_fitness, METH_VARARGS, "Evaluate GA CDC Fitness with all param in 'individual'."},
    {NULL, NULL, 0, NULL}
};

// Inizializzazione
static struct PyModuleDef cyfitnessmodule = {
    PyModuleDef_HEAD_INIT,
    "cyfitness",
    NULL,
    -1,
    CyFitnessMethods
};

PyMODINIT_FUNC PyInit_cyfitness(void){
    PyObject *m = PyModule_Create(&cyfitnessmodule);
    if(!m) return NULL;
    import_array(); // Necessario per Numpy
    return m;
}
