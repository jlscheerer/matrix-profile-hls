#!/usr/bin/env python3
import os
import struct
import argparse

# support types for the binary files
# https://docs.python.org/3/library/struct.html
types = {'double': ('d', 8, float), 'int': ('i', 4, int)}
default_type = 'double'

LogVerbose = False
LogTypeInfo, LogTypeVerbose, LogTypeError = ('INFO', 'INFO/V', 'ERROR')
def log(msg, level=LogTypeInfo):
    if level != LogTypeVerbose or LogVerbose:
        print(f'[{level}] {msg}')

def read_bin(filename, type = None, n = None):
    if type == None:
        log(f'No type specified defaulting to {default_type}', LogTypeVerbose)
        type = default_type
    if not(type in types):
        log(f'Unknown type "{type}". Type has to be one of: {", ".join(list(types))}')
        exit(-1)
    (format, length, _) = types[type]
    try:
        size = os.path.getsize(filename)
    except:
        log(f'Cannot determine size of file "{filename}"', LogTypeError)
        exit(-1)
    # attempt to automatically determine number of elements
    if n == None:
        log('Attempting to automatically determine the number of elements (n)', LogTypeVerbose)
        if size % length != 0:
            log('Failed to automatically determine the number of elements', LogTypeError)
            log(f'File contains {size} bytes, which is not a multiple of {length} bytes (type = {type})', LogTypeError)
            exit(-1)
        n = size // length
        log(f'Determined number of elements (n = {n})', LogTypeVerbose)
    if size != length * n:
        log(f'{filename} contains unexpected number of elements!', LogTypeError)
        log(f'Expected" {n} element(s) [i.e. {n * length} bytes]; but file contains {size} bytes', LogTypeError)
        exit(-1)
    try:
        with open(filename, 'rb') as file:
            data = struct.unpack(format * n, file.read())
        log(f'Read and decoded file "{filename}" ({size} bytes, i.e. {n} elements of type \'{type}\')', LogTypeVerbose)
        return data
    except:
        log(f'Cannot read file "{filename}"', LogTypeError)
        exit(-1)

def write_bin(filename, data, type):
    if not(type in types):
        log(f'Unknown type "{type}". Type has to be one of: {", ".join(list(types))}')
        exit(-1)
    (format, length, _) = types[type]
    n = len(data)
    binary = struct.pack(format * n, *data)
    try:
        with open(filename, 'wb') as file:
            file.write(binary)
    except:
        log(f'Cannot write file "{filename}"', LogTypeError)
    log(f'Wrote and encoded file "{filename}" ({length * n} bytes, i.e. {n} elements of type \'{type}\')', LogTypeInfo)
    return True

def read_ascii(filename, type = None):
    if type == None:
        log(f'No type specified defaulting to {default_type}', LogTypeVerbose)
        type = default_type
    if not(type in types):
        log(f'Unknown type "{type}". Type has to be one of: {", ".join(list(types))}')
        exit(-1)
    (format, length, fn) = types[type]
    try:
        with open(filename, 'r') as file:
            data = file.readlines()
    except:
        log(f'Cannot read file "{filename}"', LogTypeError)
        exit(-1)
    try:
        data = [line.split() for line in data]
        data = [x for x in data if len(x) > 0]
        # check if the "file format" is consistent, i.e. either all numbers in one line or all on seperate lines
        if not(len(data) == 1 or all([len(x) == 1 for x in data])):
            log('Incosistent file format detected! [Ignoring]', LogTypeVerbose)
        # flatten the list
        data = [y for x in data for y in x]
        # decode the values
        data = [fn(x) for x in data]
    except:
        log(f'Cannot decode file "{filename}"', LogTypeError)
        exit(-1)
    return data

def write_ascii(filename, data, type = None, oneline = False):
    try:
        n = len(data)
        data = (' ' if oneline else '\n').join([str(x) for x in data]) + '\n'
        with open(filename, 'w') as file:
            file.write(data)
    except:
        log(f'Cannot write file "{filename}"', LogTypeError)
        exit(-1)
    log(f'Wrote file "{filename}" ({n} elements of type \'{type}\')', LogTypeInfo)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode/decode binary time series files', usage='./ts_bin.py [OPTIONS]')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--decode', action='store_true', help='decode file from binary')
    group.add_argument('-e', '--encode', action='store_true', help='encode file to binary')
    
    parser.add_argument('input', help='the file to encode/decode', nargs=1)
    parser.add_argument('-o', '--output', help='specify the output file name')
    
    parser.add_argument('-t', '--type', choices=list(types), help='type of numbers')
    
    parser.add_argument('-n', type=int, help='count of numbers to be decode [if -n is not specified value will be inferred]')
    
    parser.add_argument('-l', '--limit', type=int, help='limit the count of numbers to encode/decode')
    parser.add_argument('--offset', type=int, help='ignore the first OFFSET numbers when encoding/decoding')
    
    parser.add_argument('--oneline', action='store_true', help='output decoded file in a single line')

    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')

    args = parser.parse_args()

    LogVerbose = bool(args.verbose)

    if (args.n and args.n < 0):
        parser.error('-n must have a non-negative value')

    if (args.limit and args.limit < 0):
        parser.error('-l/--limit must have a non-negative value')

    if (args.offset and args.offset < 0):
        parser.error('-offset must have a non-negative value')

    if args.encode:
        # encode file to binary
        if not(args.output):
            parser.error('-o/--output has to be specified in -e/--encode mode')
        if not(args.type):
            parser.error('-t/--type has to be specified in -e/--encode mode')
        if args.oneline:
            parser.error('--oneline is not applicate in -e/--encode mode')
        # read contents of the specified input file
        data = read_ascii(args.input[0], args.type)

        if args.n and len(data) != args.n:
            log('Failed to decode {args.n} numbers. Decoded {len(data)} instead [aborting]', LogTypeError)
            exit(-1)

        if args.limit or args.offset:
            offset = args.offset if args.offset else 0
            limit = (args.limit + offset if args.limit else len(data))
            try:
                log(f'Attempting to apply offset ({offset}) and/or limit ({limit - offset}) to sequence of length {len(data)}', LogTypeVerbose)
                data = data[offset:limit]
                log(f'Applied offset ({offset}) and/or limit ({limit - offset}) to sequence of length {len(data)}. Resulting sequence length: {len(data)}', LogTypeVerbose)
            except:
                log(f'Failed to apply offset ({offset}) and/or limit ({limit - offset}) to sequence of length {len(data)}', LogTypeError)
                exit(-1)

        # write to file to the specified output file
        write_bin(args.output, data, args.type)
    elif args.decode:
        # decode file from binary
        # read contents of the specified input file
        data = read_bin(args.input[0], args.type, args.n)

        if args.limit or args.offset:
            offset = args.offset if args.offset else 0
            limit = ((args.limit + offset) if args.limit else len(data))
            try:
                log(f'Attempting to apply offset ({offset}) and/or limit ({limit - offset}) to sequence of length {len(data)}', LogTypeVerbose)
                data = data[offset:limit]
                log(f'Applied offset ({offset}) and/or limit ({limit - offset}) to sequence of length {len(data)}. Resulting sequence length: {len(data)}', LogTypeVerbose)
            except:
                log(f'Failed to apply offset ({offset}) and/or limit ({limit - offset}) to sequence of length {len(data)}', LogTypeError)
                exit(-1)

        if not(args.output):
            # print the data on the console and exit
            print(*data, sep=(', ' if args.oneline else '\n'))
        else:
            # store the numbers as ascii text
            write_ascii(args.output, data, args.type, args.oneline)
    else:
        parser.error('either -d/--decode or -e/--encode has to be set')