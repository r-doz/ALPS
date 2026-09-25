import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/syncode')

# Import commonly used parts from syncode
import syncode.parsers as parsers 
from syncode import Grammar
from syncode import SyncodeLogitsProcessor
from syncode import common
from syncode.dataset import Dataset
