from argparse import ArgumentParser, Namespace

parser = ArgumentParser()

parser.usage= 'Use it kid.'
parser.add_argument('a', type = int, help =' The base value')
parser.add_argument('b', type = int, help =' The exponent')
parser.add_argument('-v', '--verbose', action ='count', help = 'Provide a verbose description. Use -vv for extra verbose')

args: Namespace = parser.parse_args()
result: int = args.a ** args.b

match args.verbose:
    case 1:
        print(f'The result is {result}')
    case 2:
        print(f'{args.a} ** {args.b} = {result}')
    case _:
        print(result)