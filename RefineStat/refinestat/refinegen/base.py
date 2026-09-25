def iterative_refine(prompt, iter_gen, checker_cls, unit_name, **kwargs):
    """
    Generalized iterative refiner that generates code iteratively while validating it using
    a provided checker class. Logs intermediate code, iteration number, and backward status
    to a file if provided, and also accumulates these log messages to return them.

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
    
    # Prepare log handling: open file if provided and also initialize an accumulator.
    log_file_path = kwargs.get('log_file', None)
    log_handle = open(log_file_path, 'w') if log_file_path is not None else None
    log_content = []  # Accumulate log messages here.

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

    while not iter_gen.finished() and iteration < max_iter and not finished_flag:
        iteration += 1
        out = iter_gen.forward(units=[unit_name], num=1)[-1]
        processed_code = process_output(prev_output, out)
        print(out)
        backward_taken = False  # Flag to indicate whether a backward step is taken.

        if processed_code is None:
            message = f"Iteration {iteration}: No new code detected"
            print(message)
            if log_handle:
                log_handle.write(message + "\n")
            log_content.append(message)
            continue

        # Clean and dedent the processed code if needed.
        if kwargs.get('dedent', True):
            processed_code = "\n".join(line.lstrip() for line in processed_code.splitlines())

        # Log intermediate code.
        message = f"Iteration {iteration}:\nIntermediate code:\n{processed_code}"
        if log_handle:
            log_handle.write(message + "\n")
        log_content.append(message)

        # Create a checker instance, passing extra config if needed.
        checker_config = kwargs.get('checker_config', {})
        checker = checker_cls(processed_code, symboltable, generator=iter_gen, unit_name=unit_name, **checker_config)
        if not checker.check():
            iter_gen.backward(unit_name)
            backward_taken = True
            message = f"Iteration {iteration}: Checker validation failed. Stepping back."
            print(message)
            if log_handle:
                log_handle.write(message + "\n")
            log_content.append(message)
            continue

        if kwargs.get('checker_type') == 'NumpyroChecker':
            if 'NUTS' in out or 'MCMC' in out or 'random' in out or 'with pyro.plate' in out:
                if 'obs' in prev_output:
                    full_log = "\n".join(log_content)
                    return prev_output, symboltable, full_log
                else:
                    iter_gen.backward(unit_name)
                    backward_taken = True
                    message = f"Iteration {iteration}: NUTS/MCMC/random found. Backward taken."
                    print(message)
                    if log_handle:
                        log_handle.write(message + "\n")
                    log_content.append(message)
                    continue
                
                
        if hasattr(checker, 'finished') and checker.finished:
            finished_flag = True

        prev_output = out
        message = f"Iteration {iteration}: Check passed. Backward taken: {backward_taken}"
        print(message)
        if log_handle:
            log_handle.write(message + "\n\n")
        log_content.append(message)

    if log_handle:
        log_handle.close()

    # Join accumulated log messages into one string.
    full_log = "\n".join(log_content)
    return out, symboltable, full_log

if __name__ == '__main__':
    print("Refer to notebook for usage examples.")
