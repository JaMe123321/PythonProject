# serve.py
import os
from waitress import serve
from app2.app import app
# 必须在其它库（numpy、sklearn 等）import 之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if __name__ == "__main__":
    serve(
        app,
        listen="*:7777",
        threads=64,
        backlog=512,
        connection_limit=200,
    )
