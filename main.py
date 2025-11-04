import runpy
import sys

def main():
    # Directly run the module instead of subprocess to show output
    sys.argv = ["scraper.twikit_collector"] + sys.argv[1:]
    runpy.run_module("scraper.twikit_collector", run_name="__main__")

if __name__ == "__main__":
    main()
