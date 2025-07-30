import socket

# 建立一個Socket，socket.SOCK_STREAM代表TCP連線
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# 綁定IP和Port，讓別人可以透過這個位置來跟server溝通
s.bind(('192.168.50.10', 9999))

# 開始監聽有沒有任何人想要來連線，也可以在此設定最多允許的連線數
s.listen(5)

# 等待其他client的請求，accept方法會阻塞，直到有client來請求為止
while True:
	(conn, client_addr) = s.accept()

	print(conn)
	print("client ip and port: ", client_addr)

	# 通信:收發訊息
	while True:
		try:
			data = conn.recv(1024) # 最大接收數據量為 1024Bytes
      
			if len(data) == 0: 
				break

			print("Client data: ", data.decode('utf-8'))
     
			conn.send(data.upper()) # 回覆client訊息
		except Exception:
			break;

	# 關閉與某個client的連接，代表與這個client的通信結束了，可以繼續等待下一個client與它建立連線
	conn.close()

  
# 關閉socket，如果關了，Client就訪問不到了
# s.close()