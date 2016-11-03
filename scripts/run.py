"""
The high-level hierarchy for running stuff. Run *this* script in the **main
directory** of the repository. Otherwise you might mess up output files.

(c) 2016 by Daniel Seita
"""

import sys
from human_player import HumanPlayer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run.py <ROM_FILE_NAME>")
        print("Note: don't cd into \'scripts\'; call it in the top-level directory")
        sys.exit()
    hp = HumanPlayer(game=sys.argv[1], 
                     rand_seed=1,
                     output_dir="output")
    hp.play_and_save()
