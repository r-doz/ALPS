# utils/helpers.py
import importlib
import inspect

def func_exist_check(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        return callable(getattr(module, function_name, None))
    except ImportError:
        return False

def params_extract(lib, func_str):
    param_names = []
    lib = importlib.import_module(lib)
    func_callable = getattr(lib, func_str)
    try:
        inherit_classes = inspect.getmro(func_callable)[:-1]
    except AttributeError:
        inherit_classes = [func_callable]
   
    for cls in inherit_classes:
        all_methods = inspect.getmembers(cls, predicate=inspect.isroutine)
        normal_methods = [name for name, func in all_methods if not (name.startswith('__') and name.endswith('__'))]
        for method in normal_methods:
            try:
                method = getattr(cls, method)
                signature = inspect.signature(method)
                param_names.extend(signature.parameters.keys())
            except Exception:
                pass
        try:
            param_names.extend(inspect.signature(cls).parameters.keys())
        except Exception:
            pass
    return list(set(param_names))



    # for cls in inherit_classes:
    #     all_methods = inspect.getmembers(cls, predicate=inspect.isroutine)
    #     normal_methods = [name for name, func in all_methods if not (name.startswith('__') and name.endswith('__'))]
    #     for method in normal_methods:
    #         method = getattr(cls, method)
    #         signature = inspect.signature(method)
    #         param_names.extend(signature.parameters.keys())
    #     try:
    #         param_names.extend(inspect.signature(cls).parameters.keys())
    #     except Exception:
    #         pass
    # return list(set(param_names))

