# refinegen/checkers/general/abstract_checker.py
import ast

class AbstractChecker:
    def __init__(self, code: str, symboltable: dict, generator=None, unit_name=None):
        """
        :param code: The code snippet to check.
        :param symboltable: A dictionary shared among checkers.
        :param generator: (Optional) The iter_gen instance for view/backward operations.
        :param unit_name: (Optional) The unit name to pass to generator.view/backward.
        """
        self.code = code
        self.symboltable = symboltable
        self.generator = generator
        self.unit_name = unit_name
        self.tree = None
        self.completion_in_progress = False  # Flag indicating code is incomplete / completion is in progress.s

    def parse(self) -> bool:
        """
        Parses the code into an AST.
        """
        try:
            self.tree = ast.parse(self.code)
            return True
        except SyntaxError as e:
            if self.generator and self.unit_name:           ## Could be that an intermediate unit has been generated but the complete unit is still being generated. Thus, we view the current unit
                try:
                    view_data = self.generator.view(self.unit_name)[0]
                    if not view_data or len(view_data) == 0:        # If at the beginning of the generation of the code.
                        print("No new view data. Marking function completion in progress.")
                        self.completion_in_progress = True
                        return True  # Instead of False, we return True to signal in-progress completion.
                    else:
                        # Try parsing the last element from the view.
                        ast.parse(view_data[-1])            ## If the last unit (intermediate unit) is not parsable, then we go back to the previous unit
                        print("Function completion in progress")
                        self.completion_in_progress = True
                        return True
                except SyntaxError:
                    self.generator.backward(self.unit_name)
                    print("Backward due to parsing issue (AbstractChecker)")
                    return False
            return False