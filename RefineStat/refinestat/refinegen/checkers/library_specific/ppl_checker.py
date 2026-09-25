# Create a new base class for PPL checkers.
from ..general.function_checker import FunctionChecker

class PplChecker(FunctionChecker):
    def __init__(self, code: str, symboltable: dict, generator=None, unit_name=None, checker_config=None):
        super().__init__(code, symboltable, generator=generator, unit_name=unit_name, checker_config=checker_config)
        self.finished = False

    def check(self) -> bool:
        # Run the generic function checks first.
        if not super().check():
            return False
        # Additional PPL-level check: if the module name is 'plt', mark the check as finished.
        if self.module_name == 'plt':
            self.finished = True
        return True


# Now update the PyMCChecker to inherit from PplChecker.
class PyMCChecker(PplChecker):
    def __init__(self, code: str, symboltable: dict, generator=None, unit_name=None, checker_config=None):
        super().__init__(code, symboltable, generator=generator, unit_name=unit_name, checker_config=checker_config)

    def check(self) -> bool:
        if not super().check():
            return False
        # PyMC-specific logic: if the function is "sample", mark as finished.
        if self.function_name == 'sample':
            self.finished = True
        return True

class NumpyroChecker(PplChecker):
    def __init__(self, code: str, symboltable: dict, generator=None, unit_name=None, checker_config=None):
        super().__init__(code, symboltable, generator=generator, unit_name=unit_name, checker_config=checker_config)

    def check(self) -> bool:
        if not super().check():
            return False

        if self.function_name == 'NUTS' or self.function_name == 'MCMC':
            self.finished = True
        return True