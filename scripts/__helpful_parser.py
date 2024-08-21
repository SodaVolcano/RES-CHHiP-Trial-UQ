import argparse
import sys
from typing import override


class HelpfulParser(argparse.ArgumentParser):
    """
    Print help message when an error occurs.
    """

    @override
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)
