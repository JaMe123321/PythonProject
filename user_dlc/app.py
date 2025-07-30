import base64,threading
import textwrap
import traceback
from flask import Flask, flash, render_template, url_for, redirect, session, render_template_string, Response, jsonify
import random
from email.message import EmailMessage
import smtplib
import pymysql
import requests
from datetime import datetime, timedelta
from ultralytics import YOLO
import cv2
from flask import Response
import csv
from io import StringIO
from flask import send_from_directory
import time
import os
import numpy as np
import urllib.parse #搭配 UTF-8 URL 編碼，為了下載的CSV檔名可為中文
import subprocess
import shlex
import socket
from flask import request, abort
from flask_socketio import SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允許所有來源
app.secret_key = "supersecretkey"  # 用於 Session 加密

#載入YOLOv模型
modelfoot = YOLO(r"D:\圖\foot\new3\train_with_earlystop22\weights\best.pt" ,'track')
modelgarbage = YOLO(r"D:\圖\garbage\all3\train18\weights\best.pt", 'track')

# 預跑一次模型，加快第一次推論速度
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
modelfoot(dummy_image)
print("✅ 腳模型 warm-up 預跑完成")
modelgarbage(dummy_image)
print("✅ 垃圾模型 warm-up 預跑完成")

# 車頭串流
CAR_URL = "rtsp://192.168.1.45:8554/live"
CAR_DIR = r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\car\records"
# 門口串流
DOOR_URL = "rtsp://192.168.1.100:8554/live"
DOOR_DIR = r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\door\records"

# pi遠端關機
ALLOWED_PIS = {'192.168.1.100','192.168.1.45'}
SHUTDOWN_PORT = 1234
cap_car  = cv2.VideoCapture( f"ffmpeg -rtsp_transport tcp -stimeout 5000000 -i {CAR_URL}", cv2.CAP_FFMPEG)
cap_door = cv2.VideoCapture( f"ffmpeg -rtsp_transport tcp -stimeout 5000000 -i {DOOR_URL}", cv2.CAP_FFMPEG)

# ─── 垃圾分類辨識狀態 ─────────────────────────────────────────────────────
level_result        = ""           # 前端顯示用：「垃圾：Tissue / Bottle / Plastic」
recognition_active  = True         # 控制 YOLO 是否啟動辨識功能
category_counts     = {            # 儲存今日各類別的統計數量
    "Tissue": 0,
    "Bottle": 0,
    "Plastic": 0,
    "Total": 0
}
latest_crop_b64     = None         # 最近一次碰線儲存的截圖（Base64 字串）
latest_crop_label   = ""           # 對應的分類標籤（文字）

saved_ids           = set()        # 存過的圖片 hash key，用來避免重複儲存

# ─── 感測器資料狀態 ─────────────────────────────────────────────────────
sensor_data         = {'data': ''} # 儲存最新感測器傳來的 JSON 字串
last_db_write_time  = 0.0          # 上次寫入 MySQL 感測器紀錄表的時間戳（節流用）

# ─── 門口人流統計 ──────────────────────────────────────────────────────
in_count            = 0            # 進場人數
out_count           = 0            # 出場人數
violation_count     = 0            # 違規人數（如未穿布鞋）
last_violation_time = ""           # 最近一次違規的時間
last_violation_labels = set()      # 當前幀內出現的違規標籤集合（避免重複）

latest_snapshot     = None         # 最近一次違規快照檔名（用於前端通知）

# ─── 系統與登入狀態 ─────────────────────────────────────────────────────
correct_password    = "40227000"   # 登入驗證密碼
reconnecting_car    = False        # 車頭鏡頭是否正在重連（避免多線程重複重連）


# 門口 snapshots 存放資料夾
os.makedirs(
    r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\door\snapshots",
    exist_ok=True
)
# 車側 captures 存放資料夾
CAPTURE_DIR = r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\car\captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# MySQL 資料庫連接設定
DB_CONFIG = {
    'host': 'localhost','port': 3306,'user': 'root','password': 'james9344',
    'db': 'rental_db','charset': 'utf8mb4','cursorclass': pymysql.cursors.DictCursor}
db = pymysql.connect(**DB_CONFIG)

# 生成隨機 6 位數 OTP
def generate_otp():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

# 發送 OTP 到使用者填寫的 Gmail
def send_otp_email(user_gmail, otp):
    sender = "jjjdddlllaaabbb@gmail.com"
    app_pw = "agyvyjwswzndghil"

    msg = EmailMessage()
    msg["Subject"] = "【OTP 驗證碼】租借系統"
    msg["From"]    = f"租借系統 <{sender}>"
    msg["To"]      = user_gmail

    html = f"""\
    <html>
      <body style="font-family: 'Noto Sans TC'; color:#333; line-height:1.5;">
        <p>親愛的使用者，您好！</p>
        <p>感謝您使用租借系統。以下是您的 <strong>OTP 驗證碼</strong>，請在 <strong>30 分鐘</strong>內完成驗證：</p>
        <h2 style="color:#1a73e8;">{otp}</h2>
        <p>若非本人操作，請忽略此郵件。<br>感謝您的配合！</p>
        <p>— STEVEN TSAI</p>

        <!-- prevent Gmail clipping: {random.randint(0,999999)} -->
      </body>
    </html>
    """
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_pw)
        server.send_message(msg)
    print("OTP 已成功發送")
    return True



# 更新 OTP 驗證邏輯
def validate_otp(entered_otp):
    try:
        with db.cursor() as cursor:
            # 查詢所有未過期且未使用的 OTP
            sql = """
            SELECT id FROM rentals 
            WHERE otp = %s AND expiration > NOW() AND used = 0
            """
            cursor.execute(sql, (entered_otp,))
            result = cursor.fetchone()

            # 如果找到符合條件的 OTP
            if result:
                # 更新該 OTP 為已使用
                update_sql = "UPDATE rentals SET used = 1 WHERE id = %s"
                cursor.execute(update_sql, (result[0],))
                db.commit()
                print(f"OTP {entered_otp} 驗證成功，已標記為使用")
                return True
            else:
                print(f"OTP {entered_otp} 驗證失敗或已過期")
                return False
    except Exception as e:
        print(f"驗證 OTP 時發生錯誤：{str(e)}")
        return False
"""
# 傳送 OTP 到 ESP32
def send_otp_to_esp32(otp):
    try:
        # 延遲載入 requests，避免啟動時載入證書庫卡住
        import requests

        esp32_ip = "192.168.1.200"  # 更新為你的 ESP32 IP 位址
        url = f"http://{esp32_ip}/send-otp"  # 對應的 API 端點
        response = requests.post(url, data={"otp": otp})

        if response.status_code == 200:
            print(f"成功發送 OTP 到 ESP32：{otp}")
            return True
        else:
            print(f"發送 OTP 失敗，狀態碼：{response.status_code}，回應內容：{response.text}")
            return False
    except Exception as e:
        print(f"錯誤：無法連線到 ESP32 - {e}")
        return False
"""

# 傳送 OTP 到 門口 Pi
def send_otp_to_pi(otp):
    pi_ip = "192.168.1.100"
    url   = f"http://{pi_ip}:5000/send-otp"
    try:
        print(f"[DEBUG] 正在呼叫 {url} ，payload={{'otp': '{otp}'}}")
        res = requests.post(url, data={"otp": otp}, timeout=5)
        print(f"[DEBUG] HTTP {res.status_code} － {res.text!r}")
        res.raise_for_status()
        return True
    except Exception as e:
        print(f"[ERROR] 發送 OTP 到 Pi 失敗：{e}")
        return False

#login路由
#跳轉路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['password'] == correct_password:  # 檢查密碼
            session['logged_in'] = True
            next_page = session.pop('next', url_for('welcome'))  # 預設導向歡迎頁(退出的話)
            return redirect(next_page)  # 成功登入後跳轉到原本想去的頁面
        else:
            error_message = "密碼錯誤，請重新輸入"  # 設定錯誤訊息
            return render_template('login.html', error=error_message)

    return render_template('login.html')

#登出路由
#跳轉路由(跳到home_redirect函式)
@app.route('/logout')
def logout():
    session.pop("logged_in", None)
    page = request.args.get("page", "home")
    print(" 使用者登出，來源頁面：", page)
    return redirect(url_for('welcome', page=page))

#主頁面路由
#跳轉路由
@app.route('/')
def root():
    # 保留 page 參數，讓 page=dashboard 不會被吃掉
    page = request.args.get('page', 'home')
    return redirect(url_for('welcome', page=page))

#welcome路由
#跳轉路由
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

#rental路由
#跳轉路由
@app.route('/rental')
def rental():
    return render_template('rental.html')

#rental路由，按下按鈕傳送資料到資料庫
@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    count = request.form['count']
    duration = request.form['duration']
    phone = request.form['phone']
    gmail = request.form['gmail']

    try:
        with db.cursor() as cursor:
            sql = """
            INSERT INTO rentals (name, count, duration, phone, gmail, is_approved, used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (name, count, duration, phone, gmail, False, False))
            db.commit()

        return render_template_string("""
            <script>
                alert("申請已送出，請等待管理者審核。");
                window.location.href = "/";
            </script>
        """)

    except Exception as e:
        print("Database error:", e)
        return render_template_string(f"""
            <script>
                alert("資料庫錯誤，請重試。");
                window.location.href = "/";
            </script>
        """)

#verify路由
#跳轉路由，初始化頁面
@app.route('/verify_requests')
def verify_requests():
    # 登入檢查
    if not session.get("logged_in"):
        session['next'] = url_for('verify_requests')
        return redirect(url_for('login'))

    # 撈取待審資料並算剩餘秒數
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute("""
                SELECT
                  id,
                  name,
                  gmail,
                  `count`,
                  duration,
                  phone,
                  TIMESTAMPDIFF(SECOND,NOW(),created_at + INTERVAL 30 MINUTE) AS remaining_seconds,
                  (otp IS NOT NULL) AS otp_sent
                FROM rentals
                WHERE created_at >= NOW() - INTERVAL 30 MINUTE
        """)
        rows = cursor.fetchall()

    # 格式化成前端需要的 list of dict
    requests_list = []
    for r in rows:
        # 轉成整數
        try:
            sec = int(r['remaining_seconds'])
        except (TypeError, ValueError):
            sec = 0
        # 負數當 0
        sec = max(sec, 0)
        requests_list.append({
            'id':                r['id'],
            'name':              r['name'],
            'gmail':             r['gmail'],
            'count':             r['count'],
            'duration':          r['duration'],
            'phone':             r['phone'],
            'remaining_seconds': sec
        })

    return render_template('verify.html', requests=requests_list)

#verify路由，通過審核時送出OTP與寫入approved_rentals資料庫
@app.route('/approve/<int:request_id>', methods=['POST'])
def approve(request_id):
    # 1. 先取出該筆申請的 Gmail
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute("SELECT gmail FROM rentals WHERE id = %s", (request_id,))
        row = cursor.fetchone()
    user_email = row['gmail'] if row else None

    # 2. 產生 OTP
    otp = generate_otp()
    print(f"[DEBUG] Generated OTP: {otp}")

    # 3. 送到 Pi
    pi_success = send_otp_to_pi(otp)
    print(f"[DEBUG] send_otp_to_pi returned: {pi_success}")

    # 4. 送到 Email（有抓到 email 才送）
    email_success = False
    if user_email:
        email_success = send_otp_email(user_email, otp)
        print(f"[DEBUG] send_otp_email returned: {email_success}")
    else:
        print("[WARN] 沒有找到該申請的 Gmail，跳過 Email 發送")

    # 5. 把這筆申請從 rentals 刪掉（或改成標記已核准）
    try:
        with db.cursor() as cursor:
            cursor.execute("DELETE FROM rentals WHERE id = %s", (request_id,))
        db.commit()
    except Exception as e:
        db.rollback()
        print("❌ 刪除 rentals 時發生錯誤：", e)

    # 6. 根據結果組提示訊息
    if pi_success and email_success:
        msg = "✅ OTP 已送至 Pi 和 Email"
    elif pi_success:
        msg = "⚠️ OTP 已送至 Pi，但 Email 發送失敗"
    elif email_success:
        msg = "⚠️ OTP 已送至 Email，但 Pi 未收到"
    else:
        msg = "❌ OTP 發送失敗"

    # 7. 回到待審頁面並跳 alert
    return render_template_string(f"""
        <script>
          alert("{msg}");
          window.location.href = "/verify_requests";
        </script>
    """)

#records路由
#查詢路由
#跳轉路由
@app.route('/records')
def show_records():
    # 登入檢查
    if not session.get("logged_in"):
        session['next'] = url_for('show_records')
        return redirect(url_for('login'))

    date_filter = request.args.get("date")

    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:  # ✅ 使用 DictCursor
            if date_filter:
                sql = """
                SELECT id, name, count, duration, phone, gmail, otp, approved_at AS created_at, expiration 
                FROM approved_rentals 
                WHERE DATE(approved_at) = %s
                """
                cursor.execute(sql, (date_filter,))
            else:
                sql = """
                SELECT id, name, count, duration, phone, gmail, otp, approved_at AS created_at, expiration 
                FROM approved_rentals
                """
                cursor.execute(sql)
            records = cursor.fetchall()

            # ✅ 格式化資料
            formatted_records = [
                {
                    "id": record["id"],
                    "name": record["name"],
                    "count": record["count"],
                    "duration": record["duration"],
                    "phone": record["phone"],
                    "gmail": record["gmail"],
                    "otp": record["otp"],
                    "created_at": str(record["created_at"]) if record["created_at"] else "",
                    "expiration": str(record["expiration"]) if record["expiration"] else ""
                }
                for record in records
            ]

        print("傳遞到模板的資料：", formatted_records)
        return render_template('records.html', records=formatted_records)

    except Exception as e:
        print("資料庫查詢錯誤：", traceback.format_exc())
        return f"資料庫查詢錯誤：{str(e)}"


#records路由
#下載路由
@app.route('/download_csv')
def download_csv():
    if not session.get("logged_in"):
        return redirect(url_for('login'))

    date_filter = request.args.get("date")

    try:
        with db.cursor() as cursor:
            if date_filter:
                cursor.execute("""
                    SELECT id, name, count, duration, phone, gmail, otp, approved_at
                    FROM approved_rentals
                    WHERE DATE(created_at) = %s
                """, (date_filter,))
            else:
                cursor.execute("""
                    SELECT id, name, count, duration, phone, gmail, otp, approved_at
                    FROM approved_rentals
                """)
            rows = cursor.fetchall()

        def generate():
            # 加上 BOM 讓 Excel 正常顯示中文
            data = ["\ufeffID,姓名,人數,時長,電話,Gmail,OTP,時間\n"]
            for row in rows:
                data.append(",".join([str(field) for field in row]) + "\n")
            return data

        filename = f"{date_filter}.csv" if date_filter else "全部紀錄.csv"
        quoted_filename = urllib.parse.quote(filename)

        return Response(generate(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_filename}"})

    except Exception as e:
        return f"匯出錯誤：{str(e)}"

#entrance路由
#初始化頁面所需資訊
#進頁面前先驗證
@app.route('/entrance')
def entrance_page():
    if not session.get("logged_in"):
        session['next'] = url_for('entrance_page')
        return redirect(url_for('login'))
    return render_template("entrance.html")

#entrance路由
#顯示鏡頭畫面
# 門口串流對外端點
@app.route('/door_feed')
def door_feed():
    return Response(
        generate_door_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_door_frames():
    global cap_door, in_count, out_count, violation_count
    global last_violation_time, latest_snapshot, reconnecting_door

    # 初始化
    in_count = out_count = violation_count = 0
    last_violation_time = None
    latest_snapshot = None
    reconnecting_door = False

    # 已 snapshot 的 ID
    snapped_ids = set()

    # 中文標籤對應
    labelfoot_map = {
        "1": "布鞋",
        "0":   "拖鞋",
    }

    prev_time = time.time()

    def reconnect_camera():
        global cap_door, reconnecting_door
        reconnecting_door = True
        try:
            print("🔁 嘗試重新連接門口鏡頭...")
            cap_door.release()
            time.sleep(1)
            cap_door = cv2.VideoCapture(DOOR_URL, cv2.CAP_FFMPEG)
            print("✅ 重連成功" if cap_door.isOpened() else "❌ 重連失敗")
        except Exception as e:
            print("❌ 重連錯誤：", e)
        reconnecting_door = False

    while True:
        # 1) 讀取 frame
        if not cap_door or not cap_door.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if not reconnecting_door:
                threading.Thread(target=reconnect_camera, daemon=True).start()
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            continue

        ret, frame = cap_door.read()
        if not ret:
            print("⚠️ 讀取失敗，黑畫面")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if not reconnecting_door:
                threading.Thread(target=reconnect_camera, daemon=True).start()
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            continue

        # 2) 計算 FPS
        now_ts = time.time()
        fps = 1.0 / (now_ts - prev_time)
        prev_time = now_ts

        # 3) YOLOv8 track
        #    stream=True：每張畫面只產生一次 results
        for results in modelfoot.track(
            frame, iou=0.3, conf=0.5, persist=True, stream=True
        ):
            # 3a) 按 track_id 存第一次違規快照
            for box in results.boxes:
                # 兼容不同版本的 track id 屬性
                track_id = None
                if hasattr(box, "id"):
                    track_id = int(box.id)
                elif hasattr(box, "tracking_id"):
                    track_id = int(box.tracking_id)
                raw_label = results.names[int(box.cls)]
                is_violation = "違規" in raw_label

                if is_violation and track_id is not None and track_id not in snapped_ids:
                    snapped_ids.add(track_id)
                    fn = f"door_violation_{track_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    path = os.path.join("static", "door", "snapshots", fn)
                    cv2.imwrite(path, frame)
                    latest_snapshot = fn
                    violation_count += 1
                    last_violation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 3b) 繪製所有框並中文化 label
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy)
                raw_label = results.names[cls]
                display_label = labelfoot_map.get(raw_label, raw_label)
                color = (0, 0, 255) if "違規" in raw_label else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{display_label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            break  # 處理完這張 frame 的 results 就跳出

        # 4) 疊加統計文字 & FPS
        h, w = frame.shape[:2]
        cv2.line(frame, (0, h//2), (w, h//2), (0,255,255), 2)
        cv2.putText(frame, f"In: {in_count}",              (10,20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Out: {out_count}",            (10,50),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.putText(frame, f"Violations: {violation_count}",(10,80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if last_violation_time:
            cv2.putText(
                frame,
                f"Last Violation: {last_violation_time}",
                (10,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 5) 編碼並送回前端
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')



#entrance路由
#儲存截圖
@app.route('/latest_snapshot')
def latest_snapshot_route():
    global latest_snapshot
    if latest_snapshot:
        # 改成你新的 snapshots 資料夾
        return send_from_directory(
            r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\door\snapshots",
            latest_snapshot,
            as_attachment=True
        )
    else:
        return "No snapshot available", 404

#entrance路由
#將前端資料傳入後台變數中
@app.route('/stats_feed')
def stats_feed():
    global in_count, out_count, violation_count, last_violation_time
    return jsonify({
        "in_count":        in_count,
        "out_count":       out_count,
        "inside_count":    in_count - out_count,
        "violation_count": violation_count,
        "violation_time":  last_violation_time or ""
    })

#entrance路由
#儲存按鈕路由
@app.route('/save_stats', methods=['POST'])
def save_stats():
    try:
        with db.cursor() as cursor:
            sql = """
                INSERT INTO entrance_stats (in_count, out_count, violation_count, violation_time)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (
                in_count,
                out_count,
                violation_count,
                last_violation_time
            ))
            db.commit()
        return "資料已成功儲存！"
    except Exception as e:
        print("儲存資料時發生錯誤：", str(e))
        return f"儲存資料失敗：{str(e)}", 500

##entrance_records路由，初始化頁面所需資訊
#門口辨識查詢資料
@app.route('/entrance_records')
def entrance_records():
    if not session.get("logged_in"):
        session['next'] = url_for('entrance_records')
        return redirect(url_for('login'))

    date_filter = request.args.get("date")

    try:
        with db.cursor() as cursor:
            if date_filter:
                sql = """
                    SELECT id, in_count, out_count, violation_count, violation_time, saved_at
                    FROM entrance_stats
                    WHERE DATE(saved_at) = %s
                    ORDER BY saved_at DESC
                """
                cursor.execute(sql, (date_filter,))
                flash(f"✅ 已載入 {date_filter} 的紀錄", "success")
            else:
                sql = """
                    SELECT id, in_count, out_count, violation_count, violation_time, saved_at
                    FROM entrance_stats
                    ORDER BY saved_at DESC
                """
                cursor.execute(sql)
                flash("✅ 已載入全部紀錄", "success")

            rows = cursor.fetchall()

        formatted = [
            {
                "id": row["id"],
                "in_count": row["in_count"],
                "out_count": row["out_count"],
                "violation_count": row["violation_count"],
                "violation_time": row["violation_time"],
                "saved_at": str(row["saved_at"])
            }
            for row in rows
        ]

        return render_template("entrance_records.html", records=formatted, date_filter=date_filter)

    except Exception as e:
        print("❌ 查詢資料發生錯誤：", type(e), e)
        flash("❌ 查詢失敗，請稍後再試", "error")
        return render_template("entrance_records.html", records=[], date_filter=date_filter)


#entrance_records路由
#門口辨識查詢資料
#下載CSV檔
@app.route('/download_entrance_csv')
def download_entrance_csv():
    if not session.get("logged_in"):
        return redirect(url_for('login'))

    date_filter = request.args.get("date")

    filename = f"{date_filter}_entrance.csv" if date_filter else "全部紀錄_entrance.csv"
    quoted_filename = urllib.parse.quote(filename)

    try:
        with db.cursor() as cursor:
            if date_filter:
                cursor.execute("""
                    SELECT id, in_count, out_count, violation_count, violation_time, saved_at
                    FROM entrance_stats
                    WHERE DATE(saved_at) = %s
                """, (date_filter,))
            else:
                cursor.execute(
                    "SELECT id, in_count, out_count, violation_count, violation_time, saved_at FROM entrance_stats")
            rows = cursor.fetchall()

        def generate():
            output = StringIO()
            writer = csv.writer(output)
            output.write('\ufeff')  # 加入 UTF-8 BOM，讓 Excel 正確顯示中文
            writer.writerow(["ID", "In Count", "Out Count", "Violation Count", "Violation Time", "Saved At"])
            for row in rows:
                writer.writerow(row)
            return output.getvalue()

        return Response(
            generate(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_filename}"}
        )

    except Exception as e:
        return f"匯出錯誤：{str(e)}"

# 前往 Car.html 頁面，進行登入檢查
@app.route('/car')
def car():
    if not session.get("logged_in"):
        session['next'] = url_for('car')  # 儲存用戶想進入的頁面
        return redirect(url_for('login'))  # 重定向到登入頁面
    return render_template('Car.html')

# 前往 Car_records.html 頁面，進行登入檢查
@app.route('/car_records')
def car_records():
    if not session.get("logged_in"):
        session['next'] = url_for('car_records')
        return redirect(url_for('login'))
    # 把 today 也传进去
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('Car_records.html', today=today)
"""取得新的資料庫連線"""
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="james9344",
        database="rental_db",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
"""將目前的垃圾分類統計寫入 sale 表
    仿照：日期、時間、總數、各類別數"""
def save_classification_to_db(counts):
    now = datetime.now()
    date_str, time_str = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
    total, t, b, p = counts['Total'], counts['Tissue'], counts['Bottle'], counts['Plastic']
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO `sale`(`date`,`time`,`Total`,`Tissue`,`Bottle`,`Plastic`)"
                " VALUES(%s,%s,%s,%s,%s,%s)",
                (date_str, time_str, total, t, b, p)
            )
            conn.commit()
    finally:
        conn.close()

"""查詢今日已存入的首筆 sale 記錄
    返回 dict：{'date', 'Total', 'Tissue', 'Bottle', 'Plastic'}"""
def get_today_sales():
    today = datetime.now().strftime("%Y-%m-%d")
    data = {'date': today, 'Total': 0, 'Tissue': 0, 'Bottle': 0, 'Plastic': 0}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM `sale` WHERE `date`=%s", (today,))
            row = cur.fetchone()
            if row:
                data.update(row)
    finally:
        conn.close()
    return data

#點擊『儲存』，把目前統計寫入 DB 並清零
@app.route('/save', methods=['POST'])
def save_data():
    global category_counts
    if category_counts['Total'] > 0:
        save_classification_to_db(category_counts)
        category_counts = {k:0 for k in category_counts}
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'no_data'})

from datetime import datetime

def save_classification_to_db(counts):
    # 1. 準備時間＆統計數值
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    total   = counts.get('Total', 0)
    tissue  = counts.get('Tissue', 0)
    bottle  = counts.get('Bottle', 0)
    plastic = counts.get('Plastic', 0)

    # 2. 建立新連線
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO `sale`
                  (`date`, `time`, `Total`, `Tissue`, `Bottle`, `Plastic`)
                VALUES
                  (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                date_str,
                time_str,
                total,
                tissue,
                bottle,
                plastic
            ))
        # 3. commit 放在 with 之外，確保 cursor 已關閉
        conn.commit()
    except Exception as e:
        # 4. 發生錯誤時 rollback 並印出
        conn.rollback()
        print(f"[DB ERROR] save_classification_to_db: {e}")
        raise
    finally:
        # 5. 一定要關閉連線
        conn.close()


@app.route('/select_date', methods=['POST'])
def select_date():
    sel = request.get_json(silent=True) or {}
    date_str = sel.get('date')

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if date_str:
                cur.execute("""
                    SELECT `date`,`time`,`Total`,`Tissue`,`Bottle`,`Plastic`
                      FROM `sale`
                     WHERE `date` = %s
                  ORDER BY `time`
                """, (date_str,))
            else:
                # 沒給 date，就抓全部
                cur.execute("""
                    SELECT `date`,`time`,`Total`,`Tissue`,`Bottle`,`Plastic`
                      FROM `sale`
                  ORDER BY `date`,`time`
                """)
            rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
        t = r['time'].strftime('%H:%M:%S')   if hasattr(r['time'], 'strftime') else str(r['time'])
        out.append({
            'date':    d,
            'time':    t,
            'Total':   int(r['Total']),
            'Tissue':  int(r['Tissue']),
            'Bottle':  int(r['Bottle']),
            'Plastic': int(r['Plastic'])
        })
    return jsonify(out), 200

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT `date`,`time`,`Total`,`Tissue`,`Bottle`,`Plastic`
                  FROM `sale`
                 WHERE `date` = %s
              ORDER BY `time`
            """, (date_str,))
            rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        # r['date'] 不会再报错了
        d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
        t = r['time'].strftime('%H:%M:%S')   if hasattr(r['time'], 'strftime') else str(r['time'])
        out.append({
            'date':    d,
            'time':    t,
            'Total':   int(r['Total']),
            'Tissue':  int(r['Tissue']),
            'Bottle':  int(r['Bottle']),
            'Plastic': int(r['Plastic'])
        })
    return jsonify(out), 200
@app.route('/time_search1')
def time_search1():
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('Car_records.html', today=today)

@app.route('/get_statistics')
def get_statistics():
    return jsonify({
        'total_gherkin': category_counts['Total'],
        'level_S':       category_counts['Tissue'],
        'level_A':       category_counts['Bottle'],
        'level_B':       category_counts['Plastic']
    })
# ─── 首頁 & 統計 API ─────────────────────────────────────────────────────
"""首頁：渲染 index.html 並帶入今日統計"""
@app.route('/')
def index():
    sales = get_today_sales()
    return render_template('Car.html',
        level_result  = level_result,
        total_gherkin = sales['Total'],
        level_S       = sales['Tissue'],
        level_A       = sales['Bottle'],
        level_B       = sales['Plastic']
    )

"""AJAX 拿前端顯示文字「垃圾：…」"""
@app.route('/get_level_result')
def get_level_result():
    return jsonify({'level_result': level_result})

"""在影像上畫出綠色方框與置中文字標籤
    box: [x1,y1,x2,y2]"""
def box_label(image, box, label='', color=(0,255,0), txt_color=(0,0,0)):
    p1 = (int(box[0]-15), int(box[1]-25))
    p2 = (int(box[2]+15), int(box[3]+25))
    cv2.rectangle(image, p1, p2, color, 4, cv2.LINE_AA)
    if label:
        center_x = (p1[0] + p2[0]) // 2
        label_y  = p1[1] - 10
        w, h      = cv2.getTextSize(label, 0, 1, 2)[0]
        lx        = center_x - w // 2
        # 背板
        cv2.rectangle(
            image,
            (lx-10,   label_y-h-10),
            (lx+w+10, label_y+10),
            color, -1, cv2.LINE_AA
        )
        cv2.putText(image, label, (lx, label_y),
                    0, 1, txt_color, 2, cv2.LINE_AA)
"""影像串流路由：回傳 multipart/x-mixed-replace 連續 jpeg
    YOLO 物件追蹤、分級計數、紅線判斷、FPS 疊加"""
@app.route('/video_feed')
def video_feed():
    def gen():
        global level_result, category_counts, latest_crop_b64, latest_crop_label, saved_ids, cap_car, reconnecting_car

        # ✅ 當 cap_car 失效時執行背景重連
        def reconnect_car():
            global cap_car, reconnecting_car
            reconnecting_car = True
            print("🔁 嘗試重新連接車頭鏡頭...")
            try:
                cap_car.release()
                time.sleep(1)
                cap_car = cv2.VideoCapture(CAR_URL, cv2.CAP_FFMPEG)
                if cap_car.isOpened():
                    print("✅ 車頭鏡頭重連成功")
                else:
                    print("❌ 車頭鏡頭仍無法開啟")
            except Exception as e:
                print("❌ 車頭重連錯誤：", e)
            reconnecting_car = False

        # 📐 取得影像大小與中線
        width = int(cap_car.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_car.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)

        crossed_ids, touched_ids = set(), set()
        last_seen, prev_y2 = {}, {}
        start, frame_count = time.time(), 0
        threshold_y = size[1] - 350
        label_map = {0: "Bottle", 1: "Tissue", 2: "Plastic"}

        while recognition_active:
            # ✅ 若裝置異常，回傳黑畫面並啟動 thread 嘗試重連
            if not cap_car or not cap_car.isOpened():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if not reconnecting_car:
                    threading.Thread(target=reconnect_car, daemon=True).start()
                _, buf = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                continue

            # ✅ 擷取車頭鏡頭畫面
            ret, frame = cap_car.read()
            if not ret or frame is None:
                print("⚠️ 車頭鏡頭讀取失敗，顯示黑畫面並嘗試重連")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if not reconnecting_car:
                    threading.Thread(target=reconnect_car, daemon=True).start()
                _, buf = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                continue

            disp = frame.copy()  # 建立一份畫面副本，用來畫框與顯示
            results = modelgarbage.track(frame, iou=0.3, conf=0.5, persist=True, device="cuda")  # 使用 YOLO 模型進行物件追蹤
            now = time.time()

            # 移除超過 5 秒未出現的 tracking ID
            for tid, t0 in list(last_seen.items()):
                if now - t0 > 5:
                    last_seen.pop(tid)
                    crossed_ids.discard(tid)
                    touched_ids.discard(tid)
                    prev_y2.pop(tid, None)

            for box in results[0].boxes:  # 遍歷 YOLO 模型偵測出的每個物件框
                cls = int(box.cls[0])                         # 取得類別編號
                label = label_map.get(cls, "未知")             # 查表轉成文字標籤
                level_result = label                          # 更新目前全域辨識結果

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())           # 解析出 bounding box 座標
                tid = int(box.id.item()) if box.id is not None else None  # 若有 ID，則轉為整數

                if tid is not None:
                    last_seen[tid] = last_seen.get(tid, now)

                    # 判斷是否觸線，且該 ID 尚未被擷取過圖
                    if tid and tid not in touched_ids and y1 <= threshold_y <= y2:
                        appear_duration = now - last_seen.get(tid, now)
                        if appear_duration >= 1.5:  # 若存在超過 1.5 秒才儲存
                            crop = frame[y1:y2, x1:x2]
                            ok, buf = cv2.imencode('.jpg', crop)
                            if ok:
                                # 產生唯一 hash key 避免重複
                                crop_hash = f"{label}_{x1}_{y1}_{x2}_{y2}"
                                if crop_hash not in saved_ids:
                                    touched_ids.add(tid)
                                    saved_ids.add(crop_hash)

                                    latest_crop_b64 = base64.b64encode(buf).decode()
                                    latest_crop_label = label

                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"{label}_id{tid}_{ts}.jpg"
                                    filepath = os.path.join(CAPTURE_DIR, filename)
                                    with open(filepath, 'wb') as f:
                                        f.write(buf.tobytes())
                                    print(f"[SAVE] 已存檔：{filepath}")

                # 判斷物件是否從紅線上方穿越到下方
                if tid is not None:
                    prev = prev_y2.get(tid, 0)
                    if prev < threshold_y <= y2 and tid not in crossed_ids:
                        crossed_ids.add(tid)                 # 標記已穿越
                        category_counts[label] += 1          # 該類別累加計數
                        category_counts["Total"] += 1        # 總數加 1
                    prev_y2[tid] = y2                        # 更新該 ID 的 y2 座標

                # 在畫面中畫出 bounding box 與標籤
                box_label(disp, box.xyxy[0], label)

            # 畫出紅線（判斷基準線）與即時計算 FPS 顯示
            cv2.line(disp, (0, threshold_y), (disp.shape[1], threshold_y),
                     (0, 0, 255), 2, cv2.LINE_AA)
            frame_count += 1
            elapsed = now - start
            if elapsed > 0:#<!--  -->
                fps = frame_count / elapsed
                cv2.putText(disp, f"FPS: {fps:.2f}",
                            (disp.shape[1] - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

            # 將畫面壓縮為 JPEG 並透過 yield 回傳給瀏覽器串流顯示
            ok2, buf2 = cv2.imencode('.jpg', disp)
            if ok2:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf2.tobytes() + b'\r\n')

    # 使用 multipart 格式回傳 JPEG 影像流，讓前端 <img> 可以即時顯示
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_snapshot2')
def latest_snapshot2():
    return jsonify({
        'label': latest_crop_label,
        'image': latest_crop_b64
    })

# ─── 門口自動錄影程式 ─────────────────────────────────────────────────────────
#  持續錄製 RTSP 串流，並依照日期自動切換資料夾存檔： - 錄影檔存於 records/YYYY-MM-DD/rec_HHMMSS.mp4

def ffmpeg_record_loop(rtsp_url: str, base_dir: str, segment_length: int = 180):
    # 確保根目錄存在
    os.makedirs(base_dir, exist_ok=True)

    while True:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        # 每天一個子資料夾
        day_dir = os.path.join(base_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)

        time_str = now.strftime("%H%M%S")
        output_path = os.path.join(day_dir, f"rec_{time_str}.mp4")

        # 計算錄影長度（段長 or 到半夜為止）
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        duration = min(segment_length, (next_midnight - now).total_seconds())

        cmd = (
            f"ffmpeg -rtsp_transport tcp "
            f"-i {shlex.quote(rtsp_url)} "
            f"-c copy "
            f"-t {int(duration)} "
            f"{shlex.quote(output_path)}"
        )
        print(f"🔴 開始錄影：{output_path} （{int(duration)} 秒）")
        subprocess.run(shlex.split(cmd),
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)

@app.route('/download_car_csv')
def download_car_csv():
    if not session.get("logged_in"):
        return redirect(url_for('login'))

    # 不帶 date 參數，就是下載全部
    filename = "全部紀錄_car.csv"
    quoted_filename = urllib.parse.quote(filename)

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT `date`, `time`, `Total`, `Tissue`, `Bottle`, `Plastic`
                FROM `sale`
                ORDER BY `date`, `time`
            """)
            rows = cursor.fetchall()
        conn.close()

        def generate():
            output = StringIO()
            writer = csv.writer(output)
            # 寫 BOM & 表頭
            output.write('\ufeff')
            writer.writerow(["日期", "時間", "總數", "面紙", "瓶子", "塑膠"])
            # 逐行寫出
            for row in rows:
                writer.writerow([
                    row["date"], row["time"],
                    row["Total"], row["Tissue"],
                    row["Bottle"], row["Plastic"]
                ])
            return output.getvalue()

        return Response(
            generate(),
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                    f"attachment; filename*=UTF-8''{quoted_filename}"
            }
        )

    except Exception as e:
        return f"匯出失敗：{e}", 500


# ─── 下載特定日期的車側統計紀錄 CSV ─────────────────────────────────
@app.route('/download_car_csv_filtered')
def download_car_csv_filtered():
    if not session.get("logged_in"):
        return redirect(url_for('login'))

    date_filter = request.args.get("date")
    filename = f"{date_filter}_car.csv" if date_filter else "全部紀錄_car.csv"
    quoted_filename = urllib.parse.quote(filename)

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            if date_filter:
                cursor.execute("""
                    SELECT `date`, `time`, `Total`, `Tissue`, `Bottle`, `Plastic`
                    FROM `sale`
                    WHERE `date` = %s
                    ORDER BY `time`
                """, (date_filter,))
            else:
                cursor.execute("""
                    SELECT `date`, `time`, `Total`, `Tissue`, `Bottle`, `Plastic`
                    FROM `sale`
                    ORDER BY `date`, `time`
                """)
            rows = cursor.fetchall()
        conn.close()

        def generate():
            output = StringIO()
            writer = csv.writer(output)
            output.write('\ufeff')
            writer.writerow(["日期", "時間", "總數量", "衛生紙", "寶特瓶", "塑膠袋"])
            for row in rows:
                writer.writerow([
                    row["date"], row["time"],
                    row["Total"], row["Tissue"],
                    row["Bottle"], row["Plastic"]
                ])
            return output.getvalue()

        return Response(
            generate(),
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                    f"attachment; filename*=UTF-8''{quoted_filename}"
            }
        )

    except Exception as e:
        return f"匯出失敗：{e}", 500

# ─── 感測器頁面 & 資料 API ───────────────────────────────────────────────
"""接收外部感測器 POST JSON：
    - 永遠存到 sensor_data 讓前端 /sensor_data 可讀
    - 拆解成 5 組距離，若有小於 30 且距離上次寫入 >1s，寫入 sensor_log"""
@app.route('/data', methods=['POST'])
def receive_data():
    global sensor_data, last_db_write_time

    # 1. 取得原始 JSON
    sensor_data = request.get_json(force=True) or {}

    raw   = sensor_data.get('data', '')
    parts = raw.split(',')
    if len(parts) == 5:
        try:
            left, fl, f, fr, right = map(float, parts)
        except ValueError:
            print("[DATA ERROR] parse float fail:", parts)
        else:
            now_ts = time.time()
            # 2. 若任意距離<30 且已超過1秒，寫入 DB
            if any(d < 30.0 for d in (left, fl, f, fr, right)) \
               and (now_ts - last_db_write_time) > 1.0:

                conn = get_db_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                          INSERT INTO `sensor_log`
                            (`timestamp`,`left_dist`,`front_left_dist`,
                             `front_dist`,`front_right_dist`,`right_dist`)
                          VALUES (NOW(), %s, %s, %s, %s, %s)
                        """, (left, fl, f, fr, right))
                    conn.commit()
                    print(f"[DB] 記錄感測器：{parts}")
                except Exception as e:
                    print("[DB ERROR]", e)
                finally:
                    conn.close()

                last_db_write_time = now_ts
            else:
                print(f"[SKIP] 條件不符或 1s 未到：{parts}")
    else:
        print("[DATA ERROR] 格式錯誤，parts 長度 != 5:", parts)

    # 3. 永遠回一個 ACK
    return jsonify({'status': 'ok'}), 200

@app.route('/shutdown', methods=['POST'])
def shutdown():
    data = request.get_json(silent=True) or {}
    hosts = data.get('hosts', [])
    results = {}
    PORT = 1234   # 跟樹莓派上跑的 socket server 要一致
    for ip in hosts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((ip, PORT))
                s.sendall(b'poweroff')
            results[ip] = 'OK'
        except Exception as e:
            results[ip] = f'失敗：{e}'
    return jsonify(results)

@app.route('/sensor_page')
def sensor_page():
    return render_template('sensor_page.html')

@app.route('/sensor_data')
def get_sensor_data():
    """
    解析全域的 sensor_data['data'] 字串，
    拆成 left, front_left, front, front_right, right 這五個欄位後回傳。
    前端就可以直接 $.get('/sensor_data') 取到 JSON 裡的各欄位。
    """
    raw   = sensor_data.get('data', '')
    parts = raw.split(',')
    return jsonify({
        'left'        : parts[0] if len(parts) > 0 else None,
        'front_left'  : parts[1] if len(parts) > 1 else None,
        'front'       : parts[2] if len(parts) > 2 else None,
        'front_right' : parts[3] if len(parts) > 3 else None,
        'right'       : parts[4] if len(parts) > 4 else None
    }), 200

# ─── 啟動 App ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    CAR_DIR  = r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\car\records"
    DOOR_DIR = r"C:\Users\james\PycharmProjects\PythonProject\user_dlc\static\door\records"

    # 分別啟動兩條錄影 thread
    threading.Thread(
        target=lambda: ffmpeg_record_loop(CAR_URL,  CAR_DIR),
        daemon=True
    ).start()

    threading.Thread(
        target=lambda: ffmpeg_record_loop(DOOR_URL, DOOR_DIR),
        daemon=True
    ).start()

    # 最後啟動你的 Flask
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

    #代理外網
    # cd C:\ngrok
    # ngrok config add-authtoken 2zIz463knukDkq1YP1Sk27X92aK_3WWNRjTPS6pCafx4ixzkS
    # ngrok http 5000

