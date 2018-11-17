from Charm.interpreter.parser import *
from Charm.utils.charm_options import *


def main():
    ParserElement.setDefaultWhitespaceChars(' ')
    parser = get_parser()
    addCommonOptions(parser)
    addCompilerOptions(parser)
    addIOOptions(parser)
    args = parser.parse_args()
    with open(args.source, 'r') as src_file:
        src = src_file.read()
        src_file.close()

    program = Program(src, args)
    program.run(save=True)


if __name__ == '__main__':
    main()
