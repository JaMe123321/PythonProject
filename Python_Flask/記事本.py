@app.route('/search_by_date', methods=['GET'])
def search_by_date():
    try:
        date = request.args.get('date')
        connection = get_db_connection_s()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM sales WHERE date = %s", (date,))
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return jsonify(results)  # 確保這裡返回的是 JSON 格式
    except Exception as e:
        return f"Error: {e}", 500

    # Route to query sales based on year, month, and day

    #
    # @app.route('/query_sales_range', methods=['GET'])
    # def query_sales_range():
    #    start_date = request.args.get('start')
    # end_date = request.args.get('end')
    # connection = pymysql.connect(host='127.0.0.1',
    #                            port=3306,
    #                        user='root',
    #                        password='admin',
    #                        database='sales_db',
    #                        cursorclass=pymysql.cursors.DictCursor)
    # try:
    #   with connection.cursor() as cursor:
    # 更新SQL查询以聚合特定类别的销量
    #    sql = """
    #       SELECT
    #           SUM(`總數量`)AS `總數量`,
    #           SUM(`衛生紙`) AS `衛生紙`,
    #           SUM(`寶特瓶`) AS `寶特瓶`,
    #           SUM(`塑膠袋`) AS `塑膠袋`
    #       FROM
    #           sales
    #       WHERE
    #           日期 BETWEEN %s AND %s
    #       """
    #   cursor.execute(sql, (start_date, end_date))
    #   sales_data = cursor.fetchone()
    #   # 确保所有返回的值不为None，否则设置为0
    #   sales_data = {k: (v if v is not None else 0) for k, v in sales_data.items()}
    #   return jsonify(sales_data)
    # finally:
    #   connection.close()

    # 初始化分類結果統計（用來記錄各類別辨識次數）
    category_counts = {
        "衛生紙": 0,
        "寶特瓶": 0,
        "塑膠袋": 0,
        "總數量": 0
    }
    # 最新的辨識結果
    level_result = ""

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