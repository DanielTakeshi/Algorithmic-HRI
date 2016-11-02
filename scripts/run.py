"""
The high-level hierarchy for running stuff. Run *this* script.

(c) 2016 by Daniel Seita
"""

import sys
from human_player import HumanPlayer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage python [path_to_file]/run.py <ROM_FILE_NAME>")
        sys.exit()
    hp = HumanPlayer(game=sys.argv[1], 
                     rand_seed=1)
    hp.play()
