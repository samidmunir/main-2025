# Authors
# - Sami Munir | sm2246
# - Roosevelt Deves | rd939

import socket
import sys

# Create the server socket.
# function start_server(port)
# - this function will create a server socket and initialize it with
#    the local address and parameterized port number.
# - this function will be in charge of receiving data from the client
#    and returning the modified (capitalization-swapped) data back
#    to the client.
# - a while loop is infinetly run until there is no more data to receive
#    from the client. Then the function will end, and the Server.py
#    program will terminate.
def start_server(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', port))
    server_socket.listen(1)
    
    print(f'Server is listening on port: {port}.')
    
    # Accept new connections.
    connection, address = server_socket.accept()
    print(f'\tConnection from {address}...')
    
    while True:
        # Receive data from the client.
        data = connection.recv(1024).decode()
        if not data:
            break
        
        print(f'\tReceived: {data}')
        
        # Process the data (invert the capitalization).
        data_to_uppercase = data.swapcase()
        
        # Send the updated data back to the client.
        connection.sendall(data_to_uppercase.encode())
    
    # Close the connection.
    connection.close()
    print('\n*** Connection closed ***')
    
    # Close the server socket.
    server_socket.close()
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Bad Usage: python3 Server.py PORT#')
        sys.exit(1)
    
    # Pass the PORT from argv into the start_server() function.
    port = int(sys.argv[1])
    start_server(port)