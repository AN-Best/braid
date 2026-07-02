import os
import shutil
import tempfile
import zipfile
import subprocess
import sympy as sp
from sympy import ccode

def generate_fmi3_xml(dae, model_id, guid="12345678-abcd-ef01-2345-6789abcdef01"):
    """Generates the FMI 3.0 modelDescription.xml string."""
    nx = len(dae.states)
    np_params = len(dae.params)
    nv = 2 * nx + np_params
    
    xml_header = f"""<?xml version="1.0" encoding="utf-8"?>
<fmiModelDescription
  fmiVersion="3.0"
  modelName="{model_id}"
  instantiationToken="{guid}"
  generationTool="Braid FMI 3.0 Exporter">

  <ModelExchange
    modelIdentifier="{model_id}"
    canBeInstantiatedOnlyOncePerProcess="false"
    canGetAndSetFMUstate="false"
    canSerializeFMUstate="false"/>

  <ModelVariables>"""
  
    xml_variables = []
    
    # 1. Continuous States (valueReferences: 0 to nx - 1)
    for i, state in enumerate(dae.states):
        name = state.func.__name__ if hasattr(state, "func") else str(state)
        der_vr = nx + np_params + i
        xml_variables.append(
            f'    <Float64 name="{name}" valueReference="{i}" causality="local" variability="continuous" initial="exact" start="0.0" derivative="{der_vr}" />'
        )
        
    # 2. Parameters (valueReferences: nx to nx + np - 1)
    # We retrieve default values from param_meta if available
    param_meta = getattr(dae, "param_meta", {})
    for j, param in enumerate(dae.params):
        vr = nx + j
        name = param.name
        # Find default
        default_val = 1.0
        sym_repr = sp.srepr(param)
        if sym_repr in param_meta:
            default_val = param_meta[sym_repr].get("default", 1.0)
        xml_variables.append(
            f'    <Float64 name="{name}" valueReference="{vr}" causality="parameter" variability="fixed" start="{default_val}" />'
        )
        
    # 3. Derivatives (valueReferences: nx + np to 2*nx + np - 1)
    for i, state in enumerate(dae.states):
        vr = nx + np_params + i
        name = f"der({state.func.__name__})" if hasattr(state, "func") else f"der({state})"
        xml_variables.append(
            f'    <Float64 name="{name}" valueReference="{vr}" causality="local" variability="continuous" />'
        )

    # 4. Independent Variable (time) (valueReference: 2*nx + np_params = nv)
    time_vr = 2 * nx + np_params
    xml_variables.append(
        f'    <Float64 name="time" valueReference="{time_vr}" causality="independent" variability="continuous" />'
    )
        
    xml_variables_str = "\n".join(xml_variables)
    
    # ModelStructure: define continuous states and derivatives mapping
    xml_structure_start = "\n  </ModelVariables>\n  <ModelStructure>"
    xml_structure_elements = []
    for i in range(nx):
        der_vr = nx + np_params + i
        xml_structure_elements.append(f'    <ContinuousStateDerivative valueReference="{der_vr}" />')
    
    xml_structure_str = "\n".join(xml_structure_elements)
    
    xml_footer = """
  </ModelStructure>
</fmiModelDescription>
"""
    return xml_header + "\n" + xml_variables_str + xml_structure_start + "\n" + xml_structure_str + xml_footer

def generate_fmi3_c_wrapper(dae, model_id, guid, c_path):
    """Generates the FMI 3.0 compliant C wrapper for Model Exchange."""
    nx = len(dae.states)
    np_params = len(dae.params)
    nv = 2 * nx + np_params
    
    t = dae.t
    states = dae.states
    p_symbols = dae.params
    
    # Variable mappings for substitution
    sub_dict = {}
    for i, state in enumerate(states):
        sub_dict[state] = sp.Symbol(f"y[{i}]")
    for j, param in enumerate(p_symbols):
        sub_dict[param] = sp.Symbol(f"p[{j}]")
        
    eq_strings = []
    for i, state in enumerate(states):
        state_deriv = sp.Derivative(state, t)
        if state_deriv in dae.ode_assignments:
            expr = dae.ode_assignments[state_deriv]
            expr_sub = expr.subs(sub_dict)
            expr_c = ccode(expr_sub)
            eq_strings.append(f"    ydot[{i}] = {expr_c};")
        else:
            raise ValueError(f"Derivative of state {state} is not defined in ode_assignments.")
            
    equations_code = "\n".join(eq_strings)
    
    # Default initializations for states and parameters
    default_inits = []
    # Parameters
    param_meta = getattr(dae, "param_meta", {})
    for j, param in enumerate(p_symbols):
        default_val = 1.0
        sym_repr = sp.srepr(param)
        if sym_repr in param_meta:
            default_val = param_meta[sym_repr].get("default", 1.0)
        default_inits.append(f"    comp->r[{nx + j}] = {default_val};")
        
    default_inits_code = "\n".join(default_inits)
    
    c_wrapper = f"""#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define NX {nx}
#define NP {np_params}
#define NV {nv}

// FMI 3.0 API Types
typedef void* fmi3Instance;
typedef void* fmi3InstanceEnvironment;
typedef unsigned int fmi3ValueReference;
typedef double fmi3Float64;
typedef int fmi3Boolean;
typedef int fmi3Int32;
typedef char fmi3Char;
typedef const fmi3Char* fmi3String;

typedef enum {{
    fmi3OK,
    fmi3Warning,
    fmi3Discard,
    fmi3Error,
    fmi3Fatal
}} fmi3Status;

typedef void (*fmi3LogMessageCallback)(fmi3InstanceEnvironment, fmi3Status, fmi3String, fmi3String);

// Model Instance Structure
typedef struct {{
    double r[NV]; // flat array of Float64 variables
    double time;
    char instanceName[256];
    char instantiationToken[256];
    fmi3InstanceEnvironment instanceEnvironment;
    fmi3LogMessageCallback logMessage;
}} ModelInstance;

#ifdef _WIN32
#define FMI3_Export __declspec(dllexport)
#else
#define FMI3_Export
#endif

FMI3_Export const char* fmi3GetVersion(void) {{
    return "3.0";
}}

FMI3_Export fmi3Instance fmi3InstantiateModelExchange(
    fmi3String instanceName,
    fmi3String instantiationToken,
    fmi3String resourcePath,
    fmi3Boolean visible,
    fmi3Boolean loggingOn,
    fmi3InstanceEnvironment instanceEnvironment,
    fmi3LogMessageCallback logMessage)
{{
    ModelInstance* comp = (ModelInstance*)malloc(sizeof(ModelInstance));
    if (!comp) return NULL;
    
    memset(comp->r, 0, sizeof(comp->r));
    comp->time = 0.0;
    
{default_inits_code}
    
    strncpy(comp->instanceName, instanceName, 255);
    strncpy(comp->instantiationToken, instantiationToken, 255);
    comp->instanceEnvironment = instanceEnvironment;
    comp->logMessage = logMessage;
    
    return (fmi3Instance)comp;
}}

FMI3_Export void fmi3FreeInstance(fmi3Instance instance) {{
    if (instance) {{
        free(instance);
    }}
}}

FMI3_Export fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance,
    fmi3Boolean toleranceDefined,
    fmi3Float64 tolerance,
    fmi3Float64 startTime,
    fmi3Boolean stopTimeDefined,
    fmi3Float64 stopTime)
{{
    ModelInstance* comp = (ModelInstance*)instance;
    comp->time = startTime;
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3ExitInitializationMode(fmi3Instance instance) {{
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3EnterContinuousTimeMode(fmi3Instance instance) {{
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3SetTime(fmi3Instance instance, fmi3Float64 time) {{
    ModelInstance* comp = (ModelInstance*)instance;
    comp->time = time;
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3SetContinuousStates(fmi3Instance instance, const fmi3Float64 states[], size_t nStates) {{
    ModelInstance* comp = (ModelInstance*)instance;
    for (size_t i = 0; i < nStates && i < NX; i++) {{
        comp->r[i] = states[i];
    }}
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3GetContinuousStates(fmi3Instance instance, fmi3Float64 states[], size_t nStates) {{
    ModelInstance* comp = (ModelInstance*)instance;
    for (size_t i = 0; i < nStates && i < NX; i++) {{
        states[i] = comp->r[i];
    }}
    return fmi3OK;
}}

// Compute Derivatives Function
void evaluate_derivatives(ModelInstance* comp) {{
    double t = comp->time;
    const double* y = &comp->r[0];
    const double* p = &comp->r[NX];
    double* ydot = &comp->r[NX + NP];
    
{equations_code}
}}

FMI3_Export fmi3Status fmi3GetContinuousStateDerivatives(fmi3Instance instance, fmi3Float64 derivatives[], size_t nDerivatives) {{
    ModelInstance* comp = (ModelInstance*)instance;
    evaluate_derivatives(comp);
    for (size_t i = 0; i < nDerivatives && i < NX; i++) {{
        derivatives[i] = comp->r[NX + NP + i];
    }}
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3GetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference valueReferences[],
    size_t nValueReferences,
    fmi3Float64 values[],
    size_t nValues)
{{
    ModelInstance* comp = (ModelInstance*)instance;
    evaluate_derivatives(comp);
    for (size_t i = 0; i < nValueReferences && i < nValues; i++) {{
        fmi3ValueReference vr = valueReferences[i];
        if (vr < NV) {{
            values[i] = comp->r[vr];
        }} else if (vr == NV) {{
            values[i] = comp->time;
        }} else {{
            values[i] = 0.0;
        }}
    }}
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3SetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference valueReferences[],
    size_t nValueReferences,
    const fmi3Float64 values[],
    size_t nValues)
{{
    ModelInstance* comp = (ModelInstance*)instance;
    for (size_t i = 0; i < nValueReferences && i < nValues; i++) {{
        fmi3ValueReference vr = valueReferences[i];
        if (vr < NV) {{
            comp->r[vr] = values[i];
        }} else if (vr == NV) {{
            comp->time = values[i];
        }}
    }}
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3Terminate(fmi3Instance instance) {{
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3Reset(fmi3Instance instance) {{
    ModelInstance* comp = (ModelInstance*)instance;
    memset(comp->r, 0, sizeof(comp->r));
    comp->time = 0.0;
{default_inits_code}
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3CompletedIntegratorStep(
    fmi3Instance instance,
    fmi3Boolean noSetFMUStatePriorToCurrentPoint,
    fmi3Boolean* enterEventMode,
    fmi3Boolean* terminateSimulation)
{{
    if (enterEventMode) *enterEventMode = 0;
    if (terminateSimulation) *terminateSimulation = 0;
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3EnterEventMode(
    fmi3Instance instance,
    fmi3Boolean stepEvent,
    fmi3Boolean stateEvent,
    const fmi3Int32 rootsFound[],
    size_t nRootsFound,
    fmi3Boolean timeEvent)
{{
    return fmi3OK;
}}

FMI3_Export fmi3Status fmi3UpdateDiscreteStates(
    fmi3Instance instance,
    fmi3Boolean* discreteStatesNeedUpdate,
    fmi3Boolean* terminateSimulation,
    fmi3Boolean* nominalsOfContinuousStatesChanged,
    fmi3Boolean* valuesOfContinuousStatesChanged,
    fmi3Boolean* nextEventTimeDefined,
    fmi3Float64* nextEventTime)
{{
    if (discreteStatesNeedUpdate) *discreteStatesNeedUpdate = 0;
    if (terminateSimulation) *terminateSimulation = 0;
    if (nominalsOfContinuousStatesChanged) *nominalsOfContinuousStatesChanged = 0;
    if (valuesOfContinuousStatesChanged) *valuesOfContinuousStatesChanged = 0;
    if (nextEventTimeDefined) *nextEventTimeDefined = 0;
    return fmi3OK;
}}

// Stubs for FMI 3.0 functions not used by Braid but required by fmpy dynamic loading
#define STUB_STATUS(name) FMI3_Export fmi3Status name() {{ return fmi3OK; }}

STUB_STATUS(fmi3InstantiateCoSimulation)
STUB_STATUS(fmi3InstantiateScheduledExecution)
STUB_STATUS(fmi3EnterConfigurationMode)
STUB_STATUS(fmi3ExitConfigurationMode)
STUB_STATUS(fmi3EnterStepMode)
STUB_STATUS(fmi3ActivateModelPartition)
STUB_STATUS(fmi3DoStep)
STUB_STATUS(fmi3GetEventIndicators)
STUB_STATUS(fmi3GetNumberOfEventIndicators)
STUB_STATUS(fmi3GetNumberOfContinuousStates)
FMI3_Export fmi3Status fmi3GetNominalsOfContinuousStates(fmi3Instance instance, fmi3Float64 nominals[], size_t nContinuousStates) {{
    for (size_t i = 0; i < nContinuousStates; i++) {{
        nominals[i] = 1.0;
    }}
    return fmi3OK;
}}
STUB_STATUS(fmi3GetFMUState)
STUB_STATUS(fmi3SetFMUState)
STUB_STATUS(fmi3FreeFMUState)
STUB_STATUS(fmi3SerializedFMUStateSize)
STUB_STATUS(fmi3SerializeFMUState)
STUB_STATUS(fmi3DeserializeFMUState)
STUB_STATUS(fmi3GetDirectionalDerivative)
STUB_STATUS(fmi3GetAdjointDerivative)
STUB_STATUS(fmi3GetOutputDerivatives)
STUB_STATUS(fmi3GetVariableDependencies)
STUB_STATUS(fmi3GetNumberOfVariableDependencies)
STUB_STATUS(fmi3EvaluateDiscreteStates)
STUB_STATUS(fmi3GetIntervalDecimal)
STUB_STATUS(fmi3GetIntervalFraction)
STUB_STATUS(fmi3GetShiftDecimal)
STUB_STATUS(fmi3GetShiftFraction)
STUB_STATUS(fmi3SetIntervalDecimal)
STUB_STATUS(fmi3SetIntervalFraction)
STUB_STATUS(fmi3SetShiftDecimal)
STUB_STATUS(fmi3SetShiftFraction)
STUB_STATUS(fmi3GetClock)
STUB_STATUS(fmi3SetClock)

STUB_STATUS(fmi3GetFloat32)
STUB_STATUS(fmi3SetFloat32)
STUB_STATUS(fmi3GetInt8)
STUB_STATUS(fmi3SetInt8)
STUB_STATUS(fmi3GetUInt8)
STUB_STATUS(fmi3SetUInt8)
STUB_STATUS(fmi3GetInt16)
STUB_STATUS(fmi3SetInt16)
STUB_STATUS(fmi3GetUInt16)
STUB_STATUS(fmi3SetUInt16)
STUB_STATUS(fmi3GetInt32)
STUB_STATUS(fmi3SetInt32)
STUB_STATUS(fmi3GetUInt32)
STUB_STATUS(fmi3SetUInt32)
STUB_STATUS(fmi3GetInt64)
STUB_STATUS(fmi3SetInt64)
STUB_STATUS(fmi3GetUInt64)
STUB_STATUS(fmi3SetUInt64)
STUB_STATUS(fmi3GetBoolean)
STUB_STATUS(fmi3SetBoolean)
STUB_STATUS(fmi3GetString)
STUB_STATUS(fmi3SetString)
STUB_STATUS(fmi3GetBinary)
STUB_STATUS(fmi3SetBinary)
STUB_STATUS(fmi3SetDebugLogging)
"""
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(c_wrapper)

def compile_fmu_binary(c_path, dll_path):
    """Compiles the FMI wrapper and model equations into a shared library."""
    # Find compiler
    gcc_path = shutil.which("gcc") or shutil.which("x86_64-w64-mingw32-gcc")
    if gcc_path:
        cmd = [gcc_path, "-shared", "-O3", "-o", dll_path, c_path]
        subprocess.check_call(cmd)
        return True
        
    clang_path = shutil.which("clang")
    if clang_path:
        cmd = [clang_path, "-shared", "-O3", "-o", dll_path, c_path]
        subprocess.check_call(cmd)
        return True
        
    cl_path = shutil.which("cl")
    if cl_path:
        cmd = [cl_path, "/LD", "/O2", c_path, f"/Fe{dll_path}"]
        subprocess.check_call(cmd)
        return True
        
    raise RuntimeError("No compatible C compiler (gcc, clang, cl.exe) found in PATH.")

def export_fmu(dae, fmu_output_path, model_id=None):
    """Compiles the model DAE into a Functional Mock-up Unit (FMU) compliant with FMI 3.0."""
    if model_id is None:
        model_id = "BraidFMU"
        
    # We package into a temporary structure
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create FMI 3.0 modelDescription.xml
        xml_content = generate_fmi3_xml(dae, model_id)
        xml_path = os.path.join(temp_dir, "modelDescription.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
            
        # Create FMI 3.0 C wrapper
        c_path = os.path.join(temp_dir, f"{model_id}.c")
        generate_fmi3_c_wrapper(dae, model_id, "12345678-abcd-ef01-2345-6789abcdef01", c_path)
        
        # Build binary folder structure
        # FMI 3.0 Platform name for 64-bit windows is "x86_64-windows", and for linux is "x86_64-linux"
        platform_name = "x86_64-windows" if os.name == 'nt' else "x86_64-linux"
        binaries_dir = os.path.join(temp_dir, "binaries", platform_name)
        os.makedirs(binaries_dir, exist_ok=True)
        
        dll_name = f"{model_id}.dll" if os.name == 'nt' else f"{model_id}.so"
        dll_path = os.path.join(binaries_dir, dll_name)
        
        # Compile
        compile_fmu_binary(c_path, dll_path)
        
        # Package into zip archive (.fmu)
        fmu_dir = os.path.dirname(os.path.abspath(fmu_output_path))
        os.makedirs(fmu_dir, exist_ok=True)
        
        with zipfile.ZipFile(fmu_output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add xml
            zip_file.write(xml_path, "modelDescription.xml")
            # Add compiled shared library
            zip_file.write(dll_path, os.path.join("binaries", platform_name, dll_name))
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        
    return os.path.abspath(fmu_output_path)
