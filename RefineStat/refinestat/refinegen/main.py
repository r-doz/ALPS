import arviz as az
import tempfile
import os
import warnings
import traceback 
import numpy as np
import sys
from scipy.stats import kstest

# Navigate up three levels from current file, then add to path
current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
parent_dir = os.path.dirname(current_dir)  # refinegen directory
grandparent_dir = os.path.dirname(parent_dir)  # Baseline directory
sys.path.insert(0, grandparent_dir)
from commons.utils import check_model_reliability, run_pymc_code
# import pymc


def backward_till_prompt(iter_gen):
    function_call_len = len(iter_gen.view('function_call')[0])
    iter_gen.backward('function_call', function_call_len)
    symboltable = {}

def delete_pymc_sample(symboltable: dict):
    for key,value in symboltable.items():
        if 'pymc.sample' in value:
            del symboltable[key]
            break


def insert_model_code(boilerplate: str, raw_snippet) -> str:
    if isinstance(raw_snippet, list):
        raw_snippet = "".join(raw_snippet)
    model_snippet = raw_snippet.replace("```", "")
    lines = boilerplate.split('\n')
    new_lines = []
    snippet_inserted = False
    processed_snippet_lines = []
    
    # Process each line in model_snippet and stop if "pm.sample" is encountered.
    for snippet_line in model_snippet.splitlines():
        # If the line contains "pm.sample", include it and break.
        if "pm.sample" in snippet_line:
            processed_snippet_lines.append("\t" + snippet_line.lstrip())
            break
        if not snippet_line.strip():
            processed_snippet_lines.append("")
        else:
            processed_snippet_lines.append("\t" + snippet_line.lstrip())
    final_snippet = "\n".join(processed_snippet_lines)
    
    for line in lines:
        new_lines.append(line.lstrip())
        if line.strip().startswith("with pm.Model() as m:"):
            new_lines.append(final_snippet)
            snippet_inserted = True
    if not snippet_inserted:
        raise ValueError("'with pm.Model() as m:' not found in boilerplate.")
    
    return "\n".join(new_lines)


def iterative_refine(prompt, template, iter_gen, checker_cls, unit_name, **kwargs):
    """
    Generalized iterative refiner that generates code iteratively while validating it using
    a provided checker class. Logs intermediate code, iteration number, and backward status
    to a file if provided.
    
    Expected kwargs:
      - symboltable: extra symbol table entries (dict)
      - max_iter: maximum number of iterations (default 35)
      - dedent: whether to dedent the generated code (default True)
      - checker_config: extra configuration for the checker (dict)
      - log_file: path to a file where iteration logs will be saved
    """
    # Default symbol table with common mappings.
    symboltable = {
        'plt': 'matplotlib.pyplot', 
        'np': 'numpy', 
        'pd': 'pandas', 
        'az': 'arviz',
    }
    symboltable.update(kwargs.get('symboltable', {}))
    max_iter = kwargs.get('max_iter', 100)
    program_index = kwargs.get('program_index', 0)
    program_details = []
    success_program = 0
    
    # Open log file if provided.
    log_file_path = kwargs.get('log_file', None)
    log_handle = open(log_file_path, 'w') if log_file_path is not None else None

    # Helper function to process output and avoid duplicates.
    def process_output(prev, current):
        if current.startswith(prev):
            new_code = current[len(prev):]
            return new_code if new_code.strip() != "" else None
        return current

    iter_gen.start(prompt)
    iteration = 0
    prev_output = ""
    finished_flag = False
    likelihood_back = 0
    last_backtrack = 1


    while iteration< max_iter and success_program < 4:
        iteration += 1
        out = iter_gen.forward(units=[unit_name], num=1)[-1]
        processed_code = process_output(prev_output, out)
        backward_taken = False  # Flag to indicate whether a backward step is taken.

        if processed_code is None:
            if iter_gen.finished():
                backward_till_prompt(iter_gen)
            print(f"Iteration {iteration}: No new code detected")
            if log_handle:
                log_handle.write(f"Iteration {iteration}: No new code detected\n")
            continue

        # Clean and dedent the processed code if needed.
        if kwargs.get('dedent', True) and processed_code is not None:
            processed_code = "\n".join(line.lstrip() for line in processed_code.splitlines())

        # Log intermediate code.
        if log_handle:
            log_handle.write(f"Iteration {iteration}:\nIntermediate code:\n{processed_code}\n")

        # Create a checker instance, passing extra config if needed.
        checker_config = kwargs.get('checker_config', {})
        checker = checker_cls(processed_code, symboltable,  generator=iter_gen, unit_name=unit_name, **checker_config)


        if not checker.check():
            print("\n"+str(iter_gen.view(unit_name))+"\n")
            if iter_gen.view(unit_name) is not None:
                iter_gen.backward(unit_name)
                backward_taken = True
                print(f"Iteration {iteration}: Checker validation failed. Stepping back.")
                if log_handle:
                    log_handle.write(f"Iteration {iteration}: Backward taken (checker validation failed).\n")
            continue

        if hasattr(checker, 'finished') and checker.finished:
            finished_flag = True

        prev_output = out
        print(f"Iteration {iteration}: Check passed.")
        if log_handle:
            log_handle.write(f"Iteration {iteration}: Check passed. Backward taken: {backward_taken}\n\n")

        

        if iter_gen.finished() or finished_flag:
            trace_names = []
            for key,value in symboltable.items():
                if value == "pymc.sample":
                    trace_names.append(key)

            full_code = insert_model_code(template, out)
            success, ns = run_pymc_code(full_code)
            print(f"full_code for {kwargs.get('model_name')}:\n{full_code}\n")
                # print(iter_gen.view('function_call'))
            if success:
                print("Code executed successfully.")
                print(symboltable)
                print("\n")
                ## select the trace_name from the trace_names list that exist in ns
                idata = None
                trace_name_found = [key for key in trace_names if key in ns]
                if trace_name_found:
                    idata = ns[trace_name_found[0]]
                else:
                    # The incremental checker sometimes only ever reaches the
                    # `pm.sample(...)` assignment via the completion-in-progress
                    # shortcut (AbstractChecker.parse), which returns early without
                    # calling extract_call_info, so symboltable never records which
                    # variable holds the trace even though execution succeeds.
                    # Recover it directly from the executed namespace instead.
                    idata_candidates = [v for v in ns.values() if isinstance(v, az.InferenceData)]
                    if idata_candidates:
                        idata = idata_candidates[0]

                if idata is not None:
                    try:
                        cumulative_tokens = iter_gen._metadata.get('total_tokens')
                        reliability_score, diagnostics = check_model_reliability(idata)
                        program_details.append({"program": full_code,
                                                "reliability_score": reliability_score,
                                                    "diagnostics": diagnostics, "cumulative_tokens": cumulative_tokens})
                        if reliability_score > 4:
                            success_program+=1
                    except Exception as e:
                        print(f"Error in checking model reliability: {e}")
            
            current_function_call = iter_gen.view('function_call')[0]
            if len(current_function_call) == 1:
                backward_till_prompt(iter_gen)
                # A full backtrack-to-prompt resets iter_gen's own generation state
                # back to "", but prev_output (used by process_output to diff the
                # next forward() call) was never reset -- it still held the whole
                # previous program's text, so every subsequent forward() looked
                # like "no new code" (current.startswith(stale prev)) all the way
                # to max_iter, and a 2nd/3rd/4th candidate was never found.
                prev_output = ""
            ## back track to the function call that has 'observed' keyword in the function call
            elif len(current_function_call) > 1 and likelihood_back < 2:
                flag = False
                for i in range(len(current_function_call)):
                    if 'observed' in current_function_call[i]:
                        last_backtrack = len(current_function_call) - i + 1
                        delete_pymc_sample(symboltable)
                        iter_gen.backward('function_call', last_backtrack)
                        flag = True
                        likelihood_back += 1
                        break
                if not flag:
                    backward_till_prompt(iter_gen)
                    prev_output = ""
            else:
                len_function_call = len(current_function_call)
                if len_function_call > last_backtrack + 1:
                    delete_pymc_sample(symboltable)
                    iter_gen.backward('function_call', last_backtrack + 1)
                    last_backtrack += 1
                else:
                    iter_gen.backtrack_till_prompt()
                    prev_output = ""

            print("len_backtrack\n", last_backtrack)
            finished_flag = False

    if log_handle:
        log_handle.close()

    return symboltable, program_details



if __name__ == '__main__':
    print("Refer to notebook for usage examples.")
