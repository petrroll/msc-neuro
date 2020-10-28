import os, sys

class HiddenPrints:
    '''
    Allows hiding prints in jupyter notebook.

    Usage: 
        with HiddenPrints():
            print("Do not want to see this)
            call_to_something_that_prints_output_i_want_to_ignore()
    '''

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout