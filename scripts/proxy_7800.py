import socket
import threading

def forward(src, dst):
    while True:
        data = src.recv(4096)
        if not data:
            break
        dst.sendall(data)

def handle_client(client_socket):
    target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        target.connect(('127.0.0.1', 1235))
    except Exception as e:
        client_socket.close()
        return
    t1 = threading.Thread(target=forward, args=(client_socket, target), daemon=True)
    t2 = threading.Thread(target=forward, args=(target, client_socket), daemon=True)
    t1.start()
    t2.start()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 7800))
server.listen(10)
print("PROXY_LISTENING_7800", flush=True)
while True:
    client, _ = server.accept()
    threading.Thread(target=handle_client, args=(client,), daemon=True).start()
