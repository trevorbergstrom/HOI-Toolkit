import argparse

parser = argparse.ArgumentParser(description="Testing the HORCNN Model!")
parser.add_argument('--a1', help='First argument', required=False, type=int)
parser.add_argument('--a2', help='2 argument', required=True, type=int)
args = parser.parse_args()