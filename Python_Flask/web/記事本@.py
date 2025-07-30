from flask import Flask, render_template, Response, request, url_for, jsonify, redirect
import threading, time, cv2, os
from ultralytics import YOLO
from datetime import datetime
from flask_cors import CORS
import pymysql.cursors
from datetime import datetime

app = Flask(__name__)
CORS(app)
model = YOLO(r"Z:\專題\紅樓垃圾\train12\weights\best.pt", 'track')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 初始化全域變數
total_counter = 0
level_S = 0
level_A = 0
level_B = 0
level_C = 0
level_result = ""
recognition_active = True  # 全域變數控制是否持續辨識
# 初始化分類結果統計（用來記錄各類別辨識次數）
category_counts = {
    "Tissue": 0,
    "Bottle": 0,
    "Plastic": 0,
    "Total": 0
}
last_write_time = time.time()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#stream_url = 'http://192.168.52.138:8080/?action=stream'
#cap = cv2.VideoCapture(stream_url)
#cap = cv2.VideoCapture(0)
video_source = r"D:\圖\garbage\影二\3.mp4"
cap = cv2.VideoCapture(video_source)
# 檢查是否成功開啟
if not cap.isOpened():
    raise RuntimeError("❌ 無法打開攝影機，請確認是否有正確連接並未被其他程式佔用")

# 嘗試設定攝影機解析度為 1920x1080（Full HD）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

# 再次確認攝影機實際設定成功的解析度（有些攝影機不支援變更）
size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
print(f"📷 實際攝影機解析度: {size}")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 連接到資料庫sales_db
def get_db_connection_s():
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor)#查询返回字典格式
    return connection

#获当天的销售数据
def get_today_sales():
    #获取当天日期
    today_date = datetime.now().strftime("%Y-%m-%d")
    #初始化sales_data
    sales_data = {
        'date': today_date,
        'Tissue': 0,
        'Total': 0,
        'Bottle': 0,
        'Plastic': 0,
    }
    #連線資料庫
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式
    try:
        #执行 SQL 查询
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM `sale` WHERE `date` = %s", (today_date,))
            result = cursor.fetchone()
            if result:
                sales_data.update(result)
    except Exception as e:
        print("Database error:", e)
    finally:

        #關閉資料庫連結
        connection.close()

    return sales_data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 在畫面上標示邊框與標籤文字
def box_label(image, box, label='', color=(0, 255, 0), txt_color=(0, 0, 0)):
    # 計算矩形框的左上角 (p1) 和右下角 (p2)，並額外加大/縮小邊界讓框更明顯
    p1, p2 = (int(box[0] - 15), int(box[1] - 25)), (int(box[2] + 15), int(box[3] + 25))
    # 在圖像上畫出矩形框（粗線條，綠色）
    cv2.rectangle(image, p1, p2, color, thickness=4, lineType=cv2.LINE_AA)
    # 如果有標籤名稱（label），則畫出標籤背景與文字
    if label:
        # 計算標籤文字的中心位置（置中顯示）
        center_x = (p1[0] + p2[0]) // 2
        label_y = p1[1] - 10  # 標籤顯示在框上方
        # 計算文字寬高，便於定位背景框尺寸
        w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        label_x = center_x - w // 2  # 讓文字水平置中
        # 計算標籤背景框的左上角與右下角座標
        label_p1 = (label_x - 10, label_y - h - 10)
        label_p2 = (label_x + w + 10, label_y + 10)
        # 畫出標籤背景（實心矩形）
        cv2.rectangle(image, label_p1, label_p2, color, -1, cv2.LINE_AA)
        # 畫出標籤文字（黑色）
        cv2.putText(image, label, (label_x, label_y), 0, 1, txt_color, 2, cv2.LINE_AA)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 提供即時影像串流（辨識垃圾分類結果並顯示到畫面）
# ✅ 以下是已更新的 /video_feed 路由邏輯，避免重複記錄、離開後可重新記錄
@app.route('/video_feed')
def video_feed():
    def generate():
        global level_result, category_counts
        counted_ids = set()  # 已統計過的追蹤 ID
        last_seen = {}       # 每個 ID 最後一次出現時間

        start_time = time.time()
        frame_count = 0

        while recognition_active:
            label_map = {
                0: "Bottle",
                1: "Tissue",
                2: "Plastic"
            }

            ret, frame = cap.read()
            if not ret:
                continue

            results = model.track(frame, iou=0.3, conf=0.5, persist=True, device="cuda")
            boxes = results[0].boxes
            names = model.names

            now = time.time()
            # 清除 5 秒未見的 ID（讓它可重新統計）
            inactive_ids = [tid for tid, t in last_seen.items() if now - t > 5]
            for tid in inactive_ids:
                counted_ids.discard(tid)
                del last_seen[tid]

            for box in boxes:
                cls_id = int(box.cls[0])
                label = label_map.get(cls_id, "未知")
                level_result = label
                box_label(frame, box.xyxy[0], label, (0, 255, 0))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                y = y1
                track_id = int(box.id.item()) if box.id is not None else None

                print("🔍 檢查物件")
                print(f"  ➤ 類別: {label}")
                print(f"  ➤ y 座標: {y}, 門檻: {size[1] - 250}")
                print(f"  ➤ track_id: {track_id}")
                print(f"  ➤ 是否已統計: {track_id in counted_ids if track_id is not None else 'N/A'}")

                # 記錄當前 ID 的最新時間
                if track_id is not None:
                    last_seen[track_id] = now

                # 如果還沒統計過，則記錄
                if track_id is not None and track_id not in counted_ids:
                    print(f"✅ 新增統計：{label} (ID {track_id})")
                    counted_ids.add(track_id)

                    category = label_map[cls_id]
                    category_counts[category] = category_counts.get(category, 0) + 1
                    category_counts["Total"] = category_counts.get("Total", 0) + 1


            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 最新的辨識結果
    level_result = ""
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 顯示 FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示紅線（分類判定線）
    cv2.line(frame, (200, size[1] - 250), (400, size[1] - 250), (0, 0, 255), 2, 4)

      # 將圖片編碼為 JPEG 並串流回傳給前端
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#使用 pymysql 連接本地 MySQL 伺服器 (127.0.0.1)，並查詢 sales_db.sale 表中的數據
@app.route('/')
def index():
    global total_gherkin, level_S, level_A, level_B, level_C

    # 從資料庫獲取所有資料（若你前端不用這些，可以省略）
    connection = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='james9344',
        database='sales_db',
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT `date`, `Total`, `Tissue`, `Bottle`, `Plastic` FROM `sale`")
            data = cursor.fetchall()  # 如果你有用到全部資料的話
    finally:
        connection.close()

    # 抓取今日銷量數據
    sales_data = get_today_sales()
    total_gherkin = sales_data['Total']
    level_S = sales_data['Tissue']
    level_A = sales_data['Bottle']
    level_B = sales_data['Plastic']
    level_C = 0  # 若有新增分類可修改這裡

    # 回傳首頁 HTML，並傳入參數供 template 使用
    return render_template(
        'index.html',
        level_result=level_result
    )
@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global cap
    if cap and cap.isOpened():
        cap.release()
    return jsonify({
        'total_gherkin': category_counts["Total"],
        'level_S': category_counts["Tissue"],
        'level_A': category_counts["Bottle"],
        'level_B': category_counts["Plastic"],
    }), 200


@app.route('/get_level_result')
def get_level_result():
    global level_result
    return jsonify({'level_result': level_result})


@app.route('/get_statistics')
def get_statistics():
    global category_counts
    return jsonify({
        'total_gherkin': category_counts["Total"],
        'level_S': category_counts["Tissue"],
        'level_A': category_counts["Bottle"],
        'level_B': category_counts["Plastic"],
    })




@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    global length_threshold, width_threshold, curvature_threshold, area_difference_threshold

    # 直接使用表单中定义的键名来访问值
    if 'length_threshold' in request.form:
        length_threshold = float(request.form['length_threshold'])
    if 'width_threshold' in request.form:
        width_threshold = float(request.form['width_threshold'])
    if 'curvature_threshold' in request.form:
        curvature_threshold = float(request.form['curvature_threshold'])
    if 'area_difference_threshold' in request.form:
        area_difference_threshold = float(request.form['area_difference_threshold'])

    return redirect(url_for('index'))



@app.route('/save_today_settlement', methods=['POST'])
def save_today_settlement():
    # 确保已连接到数据库
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式

    data = request.json  # 从JSON获取数据
    today_date = datetime.now().strftime("%Y-%m-%d")

    try:
        with connection.cursor() as cursor:
            # 检查是否存在今天的记录
            sql = "SELECT * FROM `sales_db`.`sale` WHERE `date` = %s"
            cursor.execute(sql, (today_date,))
            existing_record = cursor.fetchone()

            if existing_record:
                # 更新记录
                sql = '''UPDATE `sale` 
                         SET `Total` = %s, `Tissue` = %s, `Bottle` = %s, `Plastic` = %s
                         WHERE `date` = %s'''
                cursor.execute(sql, (data['total_gherkin'], data['level_S'],
                                     data['level_A'], data['level_B'], today_date))
            else:
                # 插入新记录
                sql = '''INSERT INTO `sales_db`.`sale`  ( `date`, `Total`, `Tissue`, `Bottle`, `Plastic` )
                         VALUES (%s, %s, %s, %s, %s)'''
                cursor.execute(sql, (today_date, data['total_gherkin'], data['level_S'],
                                     data['level_A'], data['level_B']))
            connection.commit()
    finally:
        connection.close()

    return jsonify({'message': 'Data saved successfully!'}), 200


#@app.route('/video_feed')
#def video_feed():
  #  return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sales_query')
def sales_query():
    return render_template('sales_query.html')

@app.route('/time_search1')
def time_search1():
    # 計算今天的日期字串
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template('time_search1.html', today=today)

@app.route('/get_available_years')
def get_available_years():
    # Connect to the database
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT YEAR(`date`) AS year FROM `sales_db`.`sale` ORDER BY `year` DESC")
            years = [row['year'] for row in cursor.fetchall()]
        return jsonify(years)
    finally:
        connection.close()

@app.route('/get_available_months')
def get_available_months():
    year = request.args.get('year')
    # Connect to the database
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT MONTH(`date`) AS month FROM `sales_db`.`sale` WHERE YEAR(`date`) = %s ORDER BY `month`", (year,))
            months = [row['month'] for row in cursor.fetchall()]
        return jsonify(months)
    finally:
        connection.close()

# Route to get available days for a specific year and month from the database
@app.route('/get_available_days')
def get_available_days():
    year = request.args.get('year')
    month = request.args.get('month')
    # Connect to the database
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT DAY(`date`) AS day FROM `sales_db`.`sale` WHERE YEAR(`date`) = %s AND MONTH(`date`) = %s ORDER BY `day`", (year, month))
            days = [row['day'] for row in cursor.fetchall()]
        return jsonify(days)
    finally:
        connection.close()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 將辨識結果儲存至 MySQL 資料庫中（sales_db 資料庫的 sale 表）
total_counter = 0  # 放在全域最上方

def save_classification_to_db(counts):
    connection = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='james9344',
        db='sales_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    total = counts.get("Total", 0)
    tissue = counts.get("Tissue", 0)
    bottle = counts.get("Bottle", 0)
    plastic = counts.get("Plastic", 0)

    # 取最大值對應的主要類別並轉為 class_id
    main_class = max((k for k in ['Tissue', 'Bottle', 'Plastic']), key=lambda k: counts.get(k, 0))
    label_map_reverse = {"Bottle": 0, "Tissue": 1, "Plastic": 2}
    cls = label_map_reverse.get(main_class, 0)

    with connection.cursor() as cursor:
        sql = """
            INSERT INTO sale (`date`, `time`, `class_id`, `Total`, `Tissue`, `Bottle`, `Plastic`)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (date, time_str, cls, total, tissue, bottle, plastic))
        connection.commit()
        print(f"[✅ 寫入統計] class_id={cls}, Total={total}, Tissue={tissue}, Bottle={bottle}, Plastic={plastic}")

    connection.close()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#日期查詢
@app.route('/time_search')
def time_search():
    # 查詢資料庫，取得可查詢的年份、月份、天數等
    connection = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='james9344',
                                  db='sales_db',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT YEAR(`date`) AS year FROM `sales_db`.`sale` ORDER BY `year` DESC")
            years = [row['year'] for row in cursor.fetchall()]
    finally:
        connection.close()

    return render_template('time_search.html', years=years)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/save', methods=['POST'])
def save_data():
    global category_counts
    try:
        if category_counts["Total"] > 0:
            save_classification_to_db(category_counts)  # 你應該內部有 commit()
            print("✅ 手動寫入成功")
            category_counts = {"Total": 0, "Tissue": 0, "Bottle": 0, "Plastic": 0}
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"status": "no_data"}), 200
    except Exception as e:
        import traceback
        print("❌ 儲存失敗:", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/select_date', methods=['POST'])
def select_date():
    try:
        # 接收前端发送的 JSON 日期数据
        date = request.json.get('date')
        print(f"接收到的日期: {date}")

        # 验证日期是否为空
        if not date:
            return jsonify({"error": "日期未提供"}), 400

        # 连接数据库并查询数据
        connection = pymysql.connect(host='127.0.0.1',
                                     port=3306,
                                     user='root',
                                     password='james9344',
                                     db='sales_db',
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.DictCursor) #查询返回字典格式
        try:
            with connection.cursor() as cursor:
                # 查询指定日期的数据，從sale表格要資料
                sql = """
                    SELECT `date`, `time`, `class_id`, `Total`, `Tissue`, `Bottle`, `Plastic`
                    FROM `sale`
                    WHERE `date` = %s
                """

                #执行SQL语句cursor.execute()，并用cursor.fetchall()获取所有数据
                cursor.execute(sql, (date,))
                response_data = cursor.fetchall()

                # 格式化查询结果，設置列表formatted_data，將資料存入
                formatted_data = []
                for row in response_data:
                    formatted_row = {
                        'date': row['date'].strftime('%Y-%m-%d'),# 转换日期格式
                        'time': str(row['time']),# 转换时间格式
                        'Total': row['Total'],
                        'Tissue': row['Tissue'],
                        'Bottle': row['Bottle'],
                        'Plastic': row['Plastic']
                    }

                    formatted_data.append(formatted_row)

                print("格式化後的查询结果：", formatted_data)
        finally:
            connection.close()

        # 返回格式化的查询结果
        if formatted_data:

            # 返回 JSON 数据，HTTP 状态码 200
            return jsonify(formatted_data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        # 打印完整的错误堆栈信息
        import traceback
        #使用 traceback.format_exc() 打印完整错误信息
        print("後端發生錯誤：", traceback.format_exc())
        return jsonify({"error": "後端錯誤"}), 500
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run('0.0.0.0', port=7777, debug=True)