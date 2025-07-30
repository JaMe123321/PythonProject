import cv2
import time

rtsp_url = "rtsp://192.168.1.194:8554/live"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    raise RuntimeError("無法打開 RTSP 串流")

# 讀一張測試
ret, frame = cap.read()
if not ret:
    print("❌ 讀幀失敗")
else:
    print("✅ 讀到畫面，解析度：",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 顯示並測 FPS
start = time.time()
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("讀取中斷，重試…")
        time.sleep(0.5)
        continue
    cv2.imshow("RTSP Test", frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 每 100 幀印一次 FPS
    if count % 100 == 0:
        elapsed = time.time() - start
        print(f"FPS: {count/elapsed:.2f}")

cap.release()
cv2.destroyAllWindows()


import subprocess
import datetime
import os

RTSP_URL = "rtsp://192.168.52.70:8554/live"
os.makedirs("records", exist_ok=True)

def make_filename():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"records/rec_{ts}.mp4"

def start_recorder():
    fn = make_filename()
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", "3600",
        "-reset_timestamps", "1",
        fn
    ]
    return subprocess.Popen(cmd)

# 啟動錄影子程序
rec_proc = start_recorder()

# （你的主 loop 只跑 YOLO 影像顯示／偵測，不需要擔心錄影）
# ...

# 結束時關掉子程序
rec_proc.terminate()

@app.route('/video_feed')
def video_feed():
    def gen():
        global level_result, category_counts, latest_crop_b64, latest_crop_label

        crossed_ids, touched_ids = set(), set()
        last_seen, prev_y2 = {}, {}
        start, frame_count = time.time(), 0
        threshold_y = size[1] - 350
        label_map = {0:"Bottle",1:"Tissue",2:"Plastic"}

        while recognition_active:
            ret, frame = cap.read()
            if not ret:
                continue

            disp    = frame.copy()
            results = model.track(frame, iou=0.3, conf=0.5, persist=True, device="cuda")
            now = time.time()

            # 清理逾時的 tracking ID
            for tid, t0 in list(last_seen.items()):
                if now - t0 > 5:
                    last_seen.pop(tid)
                    crossed_ids.discard(tid)
                    touched_ids.discard(tid)
                    prev_y2.pop(tid, None)

            # 處理每個 detection box
            for box in results[0].boxes:
                cls   = int(box.cls[0])
                label = label_map.get(cls, "未知")
                level_result = label

                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                tid = int(box.id.item()) if box.id is not None else None
                if tid is not None:
                    last_seen[tid] = now

                # 碰線截圖，存 Base64
                if tid and tid not in touched_ids and y1 <= threshold_y <= y2:
                    touched_ids.add(tid)
                    crop = frame[y1:y2, x1:x2]
                    ok,buf = cv2.imencode('.jpg', crop)
                    if ok:
                        latest_crop_b64   = base64.b64encode(buf).decode()
                        latest_crop_label = label

                # 畫框 & 計數穿線
                if tid is not None:
                    prev = prev_y2.get(tid, 0)
                    if prev < threshold_y <= y2 and tid not in crossed_ids:
                        crossed_ids.add(tid)
                        category_counts[label]   += 1
                        category_counts["Total"] += 1
                    prev_y2[tid] = y2

                box_label(disp, box.xyxy[0], label)

            # 紅線 & FPS 疊加
            cv2.line(disp, (0,threshold_y), (disp.shape[1],threshold_y), (0,0,255), 2, cv2.LINE_AA)
            frame_count += 1
            elapsed = now - start
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(disp, f"FPS: {fps:.2f}",
                            (disp.shape[1]-180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2, cv2.LINE_AA)

            # 回傳 JPEG 幀
            ok, buf = cv2.imencode('.jpg', disp)
            if ok:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

< !-- templates / index.html -->
< !DOCTYPE
html >
< html
lang = "zh-Hant" >
< head >
< meta
charset = "UTF-8" / >
< !-- 放在 < head > 裡 -->
< meta
name = "viewport"
content = "width=device-width, initial-scale=1.0" >
< title > 垃圾辨識系統 < / title >
< link
rel = "stylesheet"
href = "{{ url_for('static', filename='styles.css') }}" >
< style >
*{box - sizing: border - box;
margin: 0;
padding: 0;}
body
{font - family: 'Noto Sans TC', sans - serif;
background:  # f0f2f5; color: #333; display: flex; flex-direction: column; min-height: 100vh; }

.upper - container
{display: flex;
flex - wrap: wrap;
background:  # fff; box-shadow: 0 2px 8px rgba(0,0,0,0.1); flex: 1; }
.control - area
{flex: 1 1 400px;
padding: 1.5
rem;
background: rgba(255, 255, 255, 0.9);
display: flex;
flex - direction: column;
align - items: center;
text - align: center;
gap: 1
rem;}
.control - area
h1
{font - size: 2rem;
color:  # 222; }
# current-time { font-size: 1.1rem; color: #555; }

.button - group
{display: flex;
gap: .8
rem;
flex - wrap: wrap;
justify - content: center;}
.button - common,.save - button,.sensor - btn
{
    padding: .6rem 1.2rem;
border: none;
border - radius: 6
px;
font - size: 1
rem;
font - weight: 500;
cursor: pointer;
transition: transform
.15
s;
}
.button - common.time - search1
{background:  # c0392b; color: #fff; }
.save - button
{background:  # 27ae60; color: #fff; }
.sensor - btn
{background:  # f1c40f; color: #333; }
.button - common: hover,
.save - button: hover,
.sensor - btn: hover
{transform: translateY(-2px);}
.button - common.time - search1: hover
{background:  # a93226; }
.save - button: hover
{background:  # 229954; }
.sensor - btn: hover
{background:  # d4ac0d; }

.chart - area
{width: 100 %;
height: 250
px;
margin - top: 1
rem;}
.chart - area
canvas
{width: 100 % !important;
height: 100 % !important;}

.video - area
{
    position: relative;
width: 100 %; / *滿寬 * /
                 max - width: 640
px; / *最多
640
px
寬 * /
margin: 0
auto; / *置中 * /
         / * padding - top = 高 / 寬 = 9 / 16 = 56.25 % * /
                                                padding - top: 56.25 %;
background:  # 000;
overflow: hidden;
}
.video - area
img
{
    position: absolute;
top: 0;
left: 0;
width: 100 %;
height: 100 %;
object - fit: contain; / *保留完整畫面(contain)
或裁切(cover) * /
}

.lower - container
{padding: 1.5rem;
background:  # fafafa; }
.statistics - area
{display: grid;
grid - template - columns: repeat(auto - fit, minmax(140
px, 1
fr)); gap: 1
rem;}
.statistics - item
{
    background:  # fff; border-radius:10px; padding:2rem; text-align:center; box-shadow:0 1px 6px rgba(0,0,0,0.08); transition:box-shadow .2s; }
.statistics - item: hover
{box - shadow: 0 3px 12px rgba(0, 0, 0, 0.12);}
.stat - label
{font - size: .9rem;
color:
# 777; }
.stat - value
{font - size: 2.2rem;
font - weight: 700;
color:  # 222; }

# toast {
position: fixed;
bottom: 20
px;
right: 20
px;
background: rgba(0, 0, 0, 0.75);
color:  # fff;
padding:.75
rem
1.25
rem;
border - radius: 8
px;
font - size: .9
rem;
opacity: 0;
pointer - events: none;
transition: opacity
.3
s;
z - index: 1000;
}

/ * ★ 把截圖容器移出
control - area，fixed + z - index
超高，確保在最上層 ★ * /
# capture-container {
position: fixed;
top: 520
px; / *你可以調整它想要的位置 * /
       right: 1300
px;
z - index: 9999;
display: none;
flex - direction: column;
align - items: center;
gap: .5
rem;
}
# capture-label {
color:  # fff;
background: rgba(0, 0, 0, 0.6);
padding: 4
px
8
px;
border - radius: 4
px;
font - size: .9
rem;
}
# capture-preview {
width: 200
px;
height: 200
px;
object - fit: contain;
border: 2
px
solid  # 28a745;
border - radius: 4
px;
}

/ *1）一定要有這行才會正確響應式 * /
     @ viewport
{width: device - width;}

@media(max - width

: 600
px) {
    / * 整個上半區改直排、不換行 * /
.upper - container
{
    display: flex !important;
flex - direction: column !important;
flex - wrap: nowrap !important;
align - items: stretch;
}

/ *1.
視訊區跑到最上面 * /
.video - area
{
    order: 1 !important;
width: 100 % !important;
/ *控制最高高度，超過就裁切 * /
                max - height: 200
px !important;
overflow: hidden !important;
margin - bottom: 1
rem !important;
}
.video - area
img
{
    width: 100 % !important;
height: auto !important;
object - fit: cover !important;
}

/ *2.
控制區（包含按鈕＋圖表）擺下面 * /
.control - area
{
    order: 2 !important;
width: 100 % !important;
padding: 1
rem !important;
}

/ *3.
統計卡片 * /
.lower - container
{
    display: none !important;
}
}

< / style >
    < / head >
        < body >
        < div


class ="upper-container" >

< div


class ="control-area" >

< h1 > 垃圾辨識系統 < / h1 >
< div
id = "current-time" > 目前時間：--: --:-- < / div >
< div


class ="button-group" >

< button


class ="button-common time-search1" onclick="location.href='/time_search1'" > 歷史查詢 < / button >

< button
id = "save-btn"


class ="save-button" > 儲存統計資料 < / button >

< button


class ="sensor-btn" onclick="location.href='/sensor_page'" > 即時感測器距離 < / button >

< / div >
< div


class ="chart-area" >

< canvas
id = "statsChart" > < / canvas >
< / div >
< / div >
< div


class ="video-area" >

< img
src = "{{ url_for('video_feed') }}"
alt = "video stream" / >
< / div >
< / div >

< div


class ="lower-container" >

< div


class ="statistics-area" >

< div


class ="statistics-item" >

< div


class ="stat-label" > 總數量 < / div >

< div


class ="stat-value" id="total-gherkin" > 0 < / div >

< / div >
< div


class ="statistics-item" >

< div


class ="stat-label" > 衛生紙 < / div >

< div


class ="stat-value" id="level-S" > 0 < / div >

< / div >
< div


class ="statistics-item" >

< div


class ="stat-label" > 寶特瓶 < / div >

< div


class ="stat-value" id="level-A" > 0 < / div >

< / div >
< div


class ="statistics-item" >

< div


class ="stat-label" > 塑膠袋 < / div >

< div


class ="stat-value" id="level-B" > 0 < / div >

< / div >
< / div >
< / div >

< !-- ★ 這一段放在
body
最底，並以
fixed + 高
z - index
顯示 ★ -->
< div
id = "capture-container" >
< div
id = "capture-label" > < / div >
< img
id = "capture-preview"
alt = "截圖預覽" >
< / div >

< !-- jQuery & Chart.js -->
< script
src = "https://code.jquery.com/jquery-3.6.4.min.js" > < / script >
< script
src = "https://cdn.jsdelivr.net/npm/chart.js" > < / script >
< script >
$(function(){
// 時鐘
function updateClock()
{
    const
now = new
Date(), \
    hh = now.getHours().toString().padStart(2, '0'), \
    mm = now.getMinutes().toString().padStart(2, '0'), \
    ss = now.getSeconds().toString().padStart(2, '0');
$('#current-time').text(`目前時間：${hh}:${mm}:${ss}
`);
}
updateClock();
setInterval(updateClock, 1000);

// Chart.js
const
ctx = document.getElementById('statsChart').getContext('2d'),
statsChart = new
Chart(ctx, {
    type: 'bar',
    data: {labels: ['總數量', '衛生紙', '寶特瓶', '塑膠袋'], datasets: [{label: '垃圾數量', data: [0, 0, 0, 0]}]},
    options: {scales: {y: {beginAtZero: true}}, responsive: true, maintainAspectRatio: false}
});

// 更新統計
setInterval(() = > {
$.get('/get_statistics', r= > {
$('#total-gherkin').text(r.total_gherkin);
$('#level-S').text(r.level_S);
$('#level-A').text(r.level_A);
$('#level-B').text(r.level_B);
statsChart.data.datasets[0].data = [
    r.total_gherkin, r.level_S, r.level_A, r.level_B
];
statsChart.update();
});
}, 1000);

// 儲存 & toast
$('#save-btn').click(() = > {
$.post('/save', {}, res= > {
    const
msg = res.status == = 'ok'? '✅ 儲存成功'
: res.status == = 'no_data'? '⚠️ 沒有資料可儲存' \
    : '❌ 儲存失敗';
$('#toast').text(msg).css('opacity', 1);
setTimeout(() = > $('#toast').css('opacity', 0), 1500);
}).fail(() = > {
$('#toast').text('❌ 儲存過程出錯').css('opacity', 1);
setTimeout(() = > $('#toast').css('opacity', 0), 1500);
});
});

// 左下角截圖
function
fetchLatestCapture()
{
$.get('/latest_capture', res= > {
if (res.img_b64)
{
$('#capture-preview').attr('src', 'data:image/jpeg;base64,' + res.img_b64);
$('#capture-label').text(res.label);
$('#capture-container').show();
} else {
$('#capture-container').hide();
}
});
}
setInterval(fetchLatestCapture, 1000);
});
< / script >
< !-- 放在 < / body > 前 -->
< div
id = "toast" > < / div >

< / body >
< / html >


/* 手机端：隐藏统计卡片，只留图表 */
@media (max-width: 600px) {
  /* 让 chart 区块撑满全宽 */
  .chart-area {
    width: 100% !important;
    height: 200px !important;   /* 你可以再调合适高度 */
    margin: 0 auto 1rem !important;
  }
  /* 视频全宽、自动高度，去掉黑边 */
  .video-area,
  .video-area img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    background: none !important;
    max-height: none !important;
  }
}





 @media (max-width: 600px) {
  /* ===== HEADER: 隐藏返回 & 标题一行 ===== */
  .header h2 {
    font-size: 1.3rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }
  .header a { display: none !important; }

  /* ===== VIDEO: 取消上下空隙 ===== */
  .video-area,
  .video-area img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    margin: 0 !important;
    padding: 0 !important;
  }

  /* ===== CHART: 缩短高度并去除多余边距 ===== */
  .chart-area {
    height: 160px !important;
    margin: 0 !important;
    padding: 0 !important;
  }

  /* ===== BUTTONS: 统一大小并横排 ===== */
  .button-group {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-around !important;
    gap: 0.5rem !important;
    margin: 0 auto 1rem !important;
    width: 100% !important;
    max-width: 600px;
  }
  .button-common,
  .save-button,
  .sensor-btn {
    padding: 0.6rem 1.2rem !important;
    font-size: 1rem !important;
    min-width: 0 !important;
    flex: 1 !important;
    text-align: center !important;
  }

  /* ===== 隐藏底部统计 ===== */
  .lower-container { display: none !important; }
}
  .header a {
    display: none !important;             /* 隐藏“返回主页”按钮 */
  }

  /* ===== VIDEO: 若需整体上移可调整 top ===== */
  .video-area,
  .video-area img {
     top: -3rem !important;           /* 如有需要可启用，向上移动 */
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    background: none !important;
    max-height: none !important;
  }

  /* ===== CHART: 缩低高度并调整上下间距 ===== */
  .chart-area {
    height: 220px !important;             /* 比默认小 70px */

    top: -4rem !important;
  }

  /* ===== BUTTON GROUP: 横排、缩小、向上微调 ===== */
  .button-group {
    position: relative !important;
    top: -2rem !important;                /* 向上移动 4rem (≈64px) */
    display: flex !important;
    flex-direction: row !important;       /* 横向排列 */
    justify-content: center !important;
    gap: 0.5rem !important;               /* 按钮间距 0.5rem */
    width: 100% !important;
    max-width: 600px;
    margin: 0 auto;
  }

  /* 缩小所有按钮尺寸，平分宽度 */
  .button-common,
  .save-button,
  .sensor-btn {
    padding: 0.4rem 0.8rem !important;    /* 内边距 */
    font-size: 0.9rem !important;
    border-radius: 4px !important;
    flex: 1;                               /* 平分可用空间 */
    max-width: none !important;
    text-align: center;
  }

  /* ===== HIDE LOWER CONTENT ===== */
  .lower-container {
    display: none !important;             /* 隐藏底部统计卡 */
  }

      #current-time { font-size: .9rem; }
      .video-area { max-width: 100%; }
      .chart-area { height: 180px; margin: .5rem 0; }
      .chart-area canvas { height: 100% !important; }
      .button-group {
        flex-direction: row !important;
        justify-content: space-between !important;
        gap: .5rem !important;
      }
      .button-common,
      .save-button,
      .sensor-btn {
        padding: 0.5rem 0.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        min-width: auto !important;
      }