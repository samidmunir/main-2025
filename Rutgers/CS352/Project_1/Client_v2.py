import socket as SOCK
import sys as SYS
import struct as STRUCT

if __name__ == '__main__':
    if len(SYS.argv) != 3:
        print('Usage: python3 Client.py SERVER_ADDRESS PORT')
        SYS.exit(1)
        
    SERVER_ADDRESS = SYS.argv[1]
    PORT = int(SYS.argv[2])