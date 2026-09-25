# checkers/general/function_checker.py
import ast
import builtins
from .abstract_checker import AbstractChecker
from ...utils.helpers import func_exist_check, params_extract

class FunctionChecker(AbstractChecker):
    def __init__(self, code: str, symboltable: dict, generator=None, unit_name=None, checker_config=None):
        super().__init__(code, symboltable, generator=generator, unit_name=unit_name)
        self.checker_config = checker_config or {}
        # Function call details.
        self.target_assign = None
        self.variable_name = None
        self.module_name = None
        self.function_name = None
        self.call_node = None
    def extract_call_info(self) -> bool:
        """Extracts the function call info from the AST, handling nested calls
        inside assignments and expression statements."""
        if not self.tree:
            print("AST tree is not available. Did you call parse()?")
            return False

        found = False
        call_node = None

        # Walk through all nodes to find an Assign or Expr that contains a Call
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                # Look inside the RHS for any Call
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Call):
                        self.target_assign = node
                        # call_node = child
                        self.call_node = child
                        found = True
                        break
                if found:
                    break

            elif isinstance(node, ast.Expr):
                # Look inside the Expr for any Call
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Call):
                        # self.target_assign = 
                        self.target_assign = node
                        # call_node = child
                        self.call_node = child
                        found = True
                        break
                if found:
                    break

        if not found or self.call_node is None:
            print("No function call found in the code.")
            return False

        # Determine the variable name if this was an assignment
        if isinstance(self.target_assign, ast.Assign):
            target = self.target_assign.targets[0]
            self.variable_name = target.id if isinstance(target, ast.Name) else None

        # For standalone Expr, there's no variable target
        elif isinstance(self.target_assign, ast.Expr):
            self.variable_name = None

        else:
            print("Found function call is not in an expected format.")
            return False

        # Extract module and function name from the call node
        func = self.call_node.func
        if isinstance(func, ast.Attribute):
            # e.g., pm.Normal(...)
            if isinstance(func.value, ast.Name):
                self.module_name = func.value.id
            else:
                self.module_name = None
            self.function_name = func.attr

        elif isinstance(func, ast.Name):
            # e.g., len(...)
            self.module_name = None
            self.function_name = func.id

        else:
            self.module_name = None
            self.function_name = None

        return True


    def _module_param_check(self, lib, func_str, provided_params):
        required_params = params_extract(lib, func_str)
        return all(param in required_params for param in provided_params)


    def check_function_existence(self) -> bool:
        """Checks whether the function exists in the given module."""
        # Resolve module name using symboltable if needed.

        if self.module_name is None:        # Covers the case of locally defined functions and built-ins.
             # Check if it's a built-in function.
            if hasattr(builtins, self.function_name):       # eg: len()
                return True
            # Check if it's defined locally (i.e. present in the symbol table).
            if self.function_name in self.symboltable:
                return True
            print(f"Function '{self.function_name}' not found as a built-in or in the local symbol table.")
            return False
        else:
            if self.module_name in self.symboltable:        # Resolve module name using symboltable if needed.
                self.module_name = self.symboltable[self.module_name]

            if self.module_name and self.function_name:
                if not func_exist_check(self.module_name, self.function_name):
                    print(f"Function {self.function_name} does not exist in module {self.module_name}.")
                    return False
        return True


    def check_function_arguments(self) -> bool:
        """Checks whether the function call has the expected keyword arguments."""
        # call_node = self.target_assign.value
        # call_node = self.call_node
        actual_params = [kw.arg for kw in self.call_node.keywords if kw.arg is not None]

        if self.module_name is None:
            # For built-ins or local functions, we assume argument correctness.
            return True
        else:
            if not self._module_param_check(self.module_name, self.function_name, actual_params):
                print("Argument check failed for function:", self.function_name)
                return False
        return True
    

    def check(self) -> bool:
        """Runs all generic function checks."""
        if not self.parse():        # Check-1: Parse the code.
            return False
        if self.completion_in_progress:
            return True
        if not self.extract_call_info():    # Check-2: Extract function call info.
            return False    
        if not self.check_function_existence():     # Check-3: Check if function exists in the module.
            return False
        if not self.check_function_arguments():     # Check-4: Check if function arguments are correct.
            return False

        # Update symbol table with full function identifier if applicable.
        if self.variable_name:
            full_function = f"{self.module_name}.{self.function_name}"
            # for key,value in self.symboltable.items():
            #     if value == full_function:
            #         del self.symboltable[key]
            #         break
            self.symboltable[self.variable_name] = full_function
        return True
