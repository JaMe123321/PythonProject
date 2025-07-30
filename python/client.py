import socket

# 建立一個 TCP protocol 的 socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 連到IP和port為('127.0.0.1', 9999)的server
s.connect(('192.168.50.45', 9999))


while True:
	msg = input("input the msg:")
	if len(msg) == 0: 
		continue
    
  # 如果訊息是quit => 執行break，斷開與server的連接，下次要通訊時必須重新建立連接
	if msg == 'quit':
		s.send(msg.encode('utf-8'))
		break
    
  # 傳送訊息給server
	s.send(msg.encode('utf-8'))
  
  # 接收server回傳的訊息
	data = s.recv(1024)
	print(data.decode('utf-8'))

s.close()