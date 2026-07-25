#!/usr/bin/env python3.13
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# 📍 在这里指定你要共享的目录路径
SHARED_DIR = '/Users/zhanglizhi/20260728_atguigu'  # 改成你想要的路径

authorizer = DummyAuthorizer()

# 用户 user 拥有全部权限，共享指定目录
# authorizer.add_user('user', '12345', SHARED_DIR, perm='elradfmwMT')

# 匿名用户只有读取权限（可选）
authorizer.add_anonymous(SHARED_DIR, perm='elr')

handler = FTPHandler
handler.authorizer = authorizer

server = FTPServer(('0.0.0.0', 2121), handler)
print(f"FTP 服务启动在端口 2121")
print(f"共享目录: {SHARED_DIR}")
print("账号: user / 12345")
server.serve_forever()