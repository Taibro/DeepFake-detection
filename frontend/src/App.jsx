import { useState, useEffect, useRef, useCallback } from "react";
import { Camera, Monitor, Film, RefreshCw, Play, Square, Upload, Maximize, Minimize, ExternalLink } from "lucide-react";

const API = "http://localhost:8000";

const theme = {
  huitNavy: "#003b71",         huitNavyHover: "#002b53",
  huitBlue: "#005baa",         huitRed: "#c91414",          huitRedHover: "#a60f0f",

  bgCard: "rgba(255, 255, 255, 0.94)",
  bgCardHeader: "rgba(249, 250, 251, 0.96)",
  border: "rgba(209, 213, 219, 0.8)",
  borderLight: "rgba(229, 231, 235, 0.8)",

  textDark: "#111827",         textMuted: "#4b5563",        textLight: "#9ca3af",
  successGreen: "#059669",
  successBg: "#ecfdf5",
  successBorder: "#a7f3d0",

  alertRed: "#dc2626",
  alertBg: "#fef2f2",
  alertBorder: "#fecaca",

  radius: "8px",
  shadowCard: "0 6px 24px -2px rgba(0, 0, 0, 0.09), 0 2px 6px -1px rgba(0, 0, 0, 0.05)",
  fontSans: "'Roboto', 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif",
  fontMono: "'Consolas', 'Courier New', monospace",
};

const styles = {
  appWrapper: {
    minHeight: "100vh",
    fontFamily: theme.fontSans,
    color: theme.textDark,
    display: "flex",
    flexDirection: "column",
  },
  header: {
    backgroundColor: "rgba(255, 255, 255, 0.96)",
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
    borderBottom: "2px solid #003b71",
    padding: "0 1.5rem",
    boxShadow: "0 2px 12px rgba(0, 0, 0, 0.06)",
    position: "sticky",
    top: 0,
    zIndex: 100,
  },
  headerInner: {
    maxWidth: 1360,
    margin: "0 auto",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 0",
  },
  brandGroup: {
    display: "flex",
    alignItems: "center",
    gap: 14,
  },
  logoImg: {
    width: 52,
    height: 52,
    borderRadius: "50%",
    objectFit: "contain",
    boxShadow: "0 2px 8px rgba(0, 59, 113, 0.15)",
    transition: "transform 0.25s ease",
    cursor: "pointer",
  },
  instTitle: {
    fontSize: 12,
    fontWeight: 700,
    color: theme.huitNavy,
    letterSpacing: "0.5px",
    textTransform: "uppercase",
  },
  deptTitle: {
    fontSize: 11,
    color: "#6b7280",
    fontWeight: 500,
    marginTop: 1,
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: 700,
    color: "#111827",
    marginTop: 2,
  },
  statusGroup: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    fontSize: 12,
  },
  statusPill: (isOk) => ({
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 12px",
    borderRadius: "6px",
    backgroundColor: isOk ? theme.successBg : theme.alertBg,
    color: isOk ? theme.successGreen : theme.alertRed,
    border: `1px solid ${isOk ? theme.successBorder : theme.alertBorder}`,
    fontWeight: 600,
    fontSize: 12,
  }),
  modelPill: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 12px",
    borderRadius: "6px",
    backgroundColor: "#eff6ff",
    color: theme.huitNavy,
    border: "1px solid #bfdbfe",
    fontWeight: 600,
    fontSize: 12,
  },
  main: {
    maxWidth: 1360,
    width: "100%",
    margin: "0 auto",
    padding: "1.5rem 1.5rem 3rem 1.5rem",
    flex: 1,
  },
  navBar: {
    display: "flex",
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    backdropFilter: "blur(12px)",
    borderRadius: theme.radius,
    boxShadow: "0 2px 10px rgba(0,0,0,0.06)",
    border: `1px solid ${theme.border}`,
    overflow: "hidden",
    marginBottom: "1.5rem",
  },
  navBtn: (isActive) => ({
    flex: 1,
    padding: "13px 20px",
    backgroundColor: isActive ? theme.huitNavy : "transparent",
    color: isActive ? "#ffffff" : "#374151",
    fontWeight: isActive ? 700 : 500,
    fontSize: 13.5,
    border: "none",
    borderRight: `1px solid ${theme.borderLight}`,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
  }),
  grid: {
    display: "grid",
    gap: "1.5rem",
    alignItems: "start",
  },
  card: {
    backgroundColor: theme.bgCard,
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
    border: `1px solid ${theme.border}`,
    borderRadius: theme.radius,
    boxShadow: theme.shadowCard,
    overflow: "hidden",
    transition: "box-shadow 0.25s ease, transform 0.25s ease",
  },
  cardHeader: {
    padding: "13px 18px",
    borderBottom: `1px solid ${theme.borderLight}`,
    backgroundColor: theme.bgCardHeader,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cardTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: theme.huitNavy,
    textTransform: "uppercase",
    letterSpacing: "0.4px",
  },
  cardSubtitle: {
    fontSize: 12,
    color: theme.textMuted,
    fontWeight: 500,
  },
  cardBody: {
    padding: "18px",
  },
  videoFrame: {
    backgroundColor: "#070b14",
    borderRadius: "6px",
    border: "1px solid #1f2937",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    overflow: "hidden",
    boxShadow: "inset 0 0 20px rgba(0,0,0,0.6)",
  },
  hudBadge: {
    backgroundColor: "rgba(0, 0, 0, 0.8)",
    backdropFilter: "blur(4px)",
    color: "#f9fafb",
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: 11,
    fontWeight: 500,
    fontFamily: theme.fontMono,
    border: "1px solid rgba(255,255,255,0.15)",
  },
  btnPrimary: {
    display: "inline-flex",
    alignItems: "center",
    gap: 7,
    padding: "9px 18px",
    fontSize: 13,
    fontWeight: 600,
    backgroundColor: theme.huitNavy,
    color: "#ffffff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    boxShadow: "0 2px 6px rgba(0, 59, 113, 0.25)",
    transition: "all 0.15s ease",
  },
  btnDanger: {
    display: "inline-flex",
    alignItems: "center",
    gap: 7,
    padding: "9px 18px",
    fontSize: 13,
    fontWeight: 600,
    backgroundColor: theme.huitRed,
    color: "#ffffff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    boxShadow: "0 2px 6px rgba(201, 20, 20, 0.25)",
    transition: "all 0.15s ease",
  },
  btnOutline: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "8px 14px",
    fontSize: 13,
    fontWeight: 500,
    backgroundColor: "#ffffff",
    color: "#374151",
    border: `1px solid ${theme.border}`,
    borderRadius: "5px",
    cursor: "pointer",
    transition: "all 0.15s ease",
  },
  select: {
    backgroundColor: "#ffffff",
    color: theme.textDark,
    border: `1px solid ${theme.border}`,
    padding: "8px 12px",
    fontSize: 13,
    borderRadius: "5px",
    outline: "none",
    cursor: "pointer",
    transition: "border-color 0.15s ease",
  },
  sliderContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "rgba(243, 244, 246, 0.8)",
    border: `1px solid ${theme.border}`,
    borderRadius: "6px",
    padding: "10px 8px",
    width: 36,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 12,
  },
  tableRow: {
    borderBottom: `1px solid ${theme.borderLight}`,
  },
  tableKey: {
    padding: "9px 6px",
    color: theme.textMuted,
    fontWeight: 500,
  },
  tableVal: {
    padding: "9px 6px",
    color: theme.textDark,
    fontWeight: 600,
    textAlign: "right",
  },
    footer: {
    backgroundColor: "rgba(255, 255, 255, 0.98)",
    backdropFilter: "blur(16px)",
    borderTop: "2px solid #003b71",
    marginTop: "auto",
    boxShadow: "0 -2px 12px rgba(0, 0, 0, 0.06)",
  },
  footerInner: {
    maxWidth: 1360,
    margin: "0 auto",
    padding: "2.5rem 1.5rem 1.75rem 1.5rem",
    display: "grid",
    gridTemplateColumns: "1.1fr 1.7fr 1.2fr",
    gap: "2.5rem",
  },
  footerColTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: theme.huitNavy,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: "12px",
    borderBottom: "1px solid #e5e7eb",
    paddingBottom: "6px",
  },
  footerText: {
    fontSize: 12,
    color: "#4b5563",
    lineHeight: 1.85,
  },
  footerBottomBar: {
    borderTop: "1px solid #e5e7eb",
    backgroundColor: "rgba(249, 250, 251, 0.95)",
    padding: "12px 1.5rem",
    textAlign: "center",
    fontSize: 12,
    color: "#6b7280",
  },
};

function MetricGauge({ prob, label }) {
  const isFake = prob > 0.5;
  const confidence = isFake ? prob * 100 : (1 - prob) * 100;
  return (
    <div style={{ marginTop: 12, marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
        <span style={{ color: theme.textMuted, fontWeight: 500 }}>{label}</span>
        <span style={{ color: isFake ? theme.alertRed : theme.successGreen, fontWeight: 700 }}>
          {confidence.toFixed(1)}% ({isFake ? "DEEPFAKE" : "GENUINE"})
        </span>
      </div>
      <div style={{ position: "relative", height: 10, backgroundColor: "#e5e7eb", borderRadius: "4px", overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${Math.min(100, Math.max(0, prob * 100))}%`,
            backgroundColor: isFake ? theme.huitRed : theme.successGreen,
            transition: "width 0.35s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.35s ease",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: "50%",
            width: 2,
            backgroundColor: "#374151",
            zIndex: 2,
          }}
          title="Ngưỡng phân định EER (50%)"
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: theme.textLight, marginTop: 4 }}>
        <span>0.0 (Xác thực / Real)</span>
        <span style={{ color: theme.textMuted, fontWeight: 600 }}>Ngưỡng quyết định (50%)</span>
        <span>1.0 (Giả mạo / Fake)</span>
      </div>
    </div>
  );
}

function ResultsPanel({ detections, smoothedProb, faceFound, mode }) {
  const isFake = smoothedProb > 0.5;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Kết quả Phân tích Trực tiếp</span>
          <span style={styles.cardSubtitle}>
            {faceFound ? "Đang nhận diện" : "Đang chờ tín hiệu"}
          </span>
        </div>
        <div style={styles.cardBody}>
          {faceFound ? (
            <div>
                            <div
                style={{
                  padding: "16px",
                  borderRadius: "6px",
                  backgroundColor: isFake ? theme.alertBg : theme.successBg,
                  border: `1px solid ${isFake ? theme.alertBorder : theme.successBorder}`,
                  textAlign: "center",
                  marginBottom: 14,
                  animation: "fadeIn 0.25s ease",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 700, color: isFake ? theme.alertRed : theme.successGreen, letterSpacing: "0.5px", textTransform: "uppercase" }}>
                  KẾT LUẬN GIÁM ĐỊNH
                </div>
                <div style={{ fontSize: 22, fontWeight: 800, color: isFake ? theme.alertRed : theme.successGreen, margin: "6px 0" }}>
                  {isFake ? "DEEPFAKE (GIẢ MẠO)" : "GENUINE (XÁC THỰC)"}
                </div>
                <div style={{ fontSize: 12, color: theme.textMuted }}>
                  Độ tin cậy: <strong>{(isFake ? smoothedProb * 100 : (1 - smoothedProb) * 100).toFixed(1)}%</strong> · Xác suất Deepfake: <strong>{(smoothedProb * 100).toFixed(1)}%</strong>
                </div>
              </div>

                            <MetricGauge prob={smoothedProb} label="Xác suất Deepfake Tổng hợp" />

                            <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: theme.textMuted, textTransform: "uppercase", marginBottom: 6 }}>
                  Danh sách khuôn mặt trích xuất ({detections.length})
                </div>
                <table style={styles.table}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${theme.border}`, color: theme.textLight }}>
                      <th style={{ textAlign: "left", padding: "6px 4px" }}>Đối tượng</th>
                      <th style={{ textAlign: "center", padding: "6px 4px" }}>Trạng thái</th>
                      <th style={{ textAlign: "right", padding: "6px 4px" }}>Độ tin cậy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detections.map((d, i) => (
                      <tr key={i} style={styles.tableRow}>
                        <td style={styles.tableKey}>Khuôn mặt #{i + 1}</td>
                        <td style={{ textAlign: "center", padding: "6px 4px" }}>
                          <span
                            style={{
                              display: "inline-block",
                              padding: "2px 8px",
                              borderRadius: "3px",
                              fontSize: 11,
                              fontWeight: 700,
                              backgroundColor: d.is_fake ? theme.alertBg : theme.successBg,
                              color: d.is_fake ? theme.alertRed : theme.successGreen,
                              border: `1px solid ${d.is_fake ? theme.alertBorder : theme.successBorder}`,
                            }}
                          >
                            {d.is_fake ? "FAKE" : "REAL"}
                          </span>
                        </td>
                        <td style={styles.tableVal}>{d.confidence.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "2.5rem 1rem", color: theme.textLight }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: theme.textMuted }}>Chưa phát hiện khuôn mặt</div>
              <div style={{ fontSize: 12, marginTop: 4 }}>Vui lòng điều chỉnh góc quay hoặc hướng khuôn mặt vào trung tâm</div>
            </div>
          )}
        </div>
      </div>

            <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Thông tin Mô hình</span>
        </div>
        <div style={styles.cardBody}>
          <table style={styles.table}>
            <tbody>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Mô hình đề xuất</td>
                <td style={styles.tableVal}>DeepfakeFusionModel</td>
              </tr>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Nhánh không gian</td>
                <td style={styles.tableVal}>Swin Transformer-T</td>
              </tr>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Nhánh sinh lý</td>
                <td style={styles.tableVal}>Remote-PPG (rPPG)</td>
              </tr>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Cơ chế hợp nhất</td>
                <td style={styles.tableVal}>Cross-Attention Network (CAN)</td>
              </tr>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Ngưỡng tối ưu EER</td>
                <td style={styles.tableVal}>T = 0.78 (Calibrated)</td>
              </tr>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Tập dữ liệu chuẩn</td>
                <td style={styles.tableVal}>FaceForensics++ & Celeb-DF</td>
              </tr>
              <tr style={styles.tableRow}>
                <td style={styles.tableKey}>Thiết bị suy luận</td>
                <td style={styles.tableVal}>CPU (Multi-threaded Worker)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function WebcamMode() {
  const wsRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [frame, setFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [smoothedProb, setSmoothedProb] = useState(0.5);
  const [faceFound, setFaceFound] = useState(false);
  const [fps, setFps] = useState(0);
  const fpsRef = useRef({ count: 0, last: Date.now() });

  const [cameraIndex, setCameraIndex] = useState(1);
  const [scanHeight, setScanHeight] = useState(440);
  const [isExpanded, setIsExpanded] = useState(false);

  const start = useCallback(() => {
    const ws = new WebSocket(`${API.replace("http", "ws")}/ws/webcam?camera_index=${cameraIndex}`);
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.error) { stop(); return; }
      setFrame(`data:image/jpeg;base64,${data.frame}`);
      setDetections(data.detections || []);
      setSmoothedProb(data.smoothed_prob ?? 0.5);
      setFaceFound(data.face_found ?? false);
      fpsRef.current.count++;
      const now = Date.now();
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count);
        fpsRef.current = { count: 0, last: now };
      }
    };
    ws.onerror = () => stop();
    ws.onclose = () => { setRunning(false); setFrame(null); };
    wsRef.current = ws;
    setRunning(true);
  }, [cameraIndex]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    setRunning(false); setFrame(null); setFaceFound(false); setDetections([]); setFps(0);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  return (
    <div className="tab-content" style={{ ...styles.grid, gridTemplateColumns: isExpanded ? "1fr" : "1fr 400px" }}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Luồng Giám sát Webcam</span>
          <span style={styles.cardSubtitle}>
            {running ? `Đang phát · ${fps} FPS` : "Chưa kích hoạt"}
          </span>
        </div>
        <div style={styles.cardBody}>
          <div style={{ display: "flex", gap: "10px", alignItems: "stretch" }}>
                        <div style={{ ...styles.videoFrame, height: scanHeight, flex: 1 }}>
              {!frame && <div className="scan-line" />}
              {frame ? (
                <>
                  <img src={frame} alt="Webcam Feed" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                  <div style={{ position: "absolute", top: 10, left: 10, right: 10, display: "flex", justifyContent: "space-between", pointerEvents: "none" }}>
                    <div style={styles.hudBadge}>
                      Camera: {cameraIndex === 1 ? "Webcam Rời USB (1)" : cameraIndex === 0 ? "Webcam Laptop (0)" : `Camera (${cameraIndex})`}
                    </div>
                    <div style={styles.hudBadge}>
                      Tốc độ: {fps} FPS
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: "center", color: "#9ca3af" }}>
                  <div style={{ width: 52, height: 52, borderRadius: "50%", backgroundColor: "rgba(0, 59, 113, 0.25)", border: "1px solid rgba(0, 91, 170, 0.4)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 12px" }}>
                    <Camera size={24} color="#38bdf8" />
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#f3f4f6" }}>Sẵn sàng kết nối Camera</div>
                  <div style={{ fontSize: 12, marginTop: 4, color: "#9ca3af" }}>Chọn thiết bị đầu vào bên dưới và nhấn Bắt đầu giám sát</div>
                </div>
              )}
            </div>

                        <div style={styles.sliderContainer}>
              <span style={{ fontSize: 10, fontWeight: 700, color: theme.textMuted }}>H</span>
              <input
                type="range"
                min="250"
                max="800"
                step="10"
                value={scanHeight}
                onChange={e => setScanHeight(Number(e.target.value))}
                orient="vertical"
                style={{
                  WebkitAppearance: "slider-vertical",
                  appearance: "slider-vertical",
                  width: 8,
                  flex: 1,
                  margin: "10px 0",
                  cursor: "pointer",
                }}
              />
              <span style={{ fontSize: 10, fontWeight: 600, color: theme.textMuted }}>{scanHeight}</span>
            </div>
          </div>

                    <div style={{ marginTop: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <button style={styles.btnOutline} onClick={() => setIsExpanded(!isExpanded)}>
              {isExpanded ? <><Minimize size={14} /> Thu gọn</> : <><Maximize size={14} /> Mở rộng</>}
            </button>

            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <select
                value={cameraIndex}
                onChange={e => setCameraIndex(Number(e.target.value))}
                disabled={running}
                style={{ ...styles.select, opacity: running ? 0.6 : 1 }}
              >
                <option value={1}>Camera 1: Webcam Rời USB (UGREEN 4K / Rời)</option>
                <option value={0}>Camera 0: Webcam Laptop (Tích hợp)</option>
                <option value={2}>Camera 2: Webcam Rời USB (Kênh phụ)</option>
                <option value={3}>Camera 3: OBS Virtual Camera / Khác</option>
              </select>

              {!running ? (
                <button style={styles.btnPrimary} onClick={start}>
                  <Play size={14} /> Bắt đầu giám sát
                </button>
              ) : (
                <button style={styles.btnDanger} onClick={stop}>
                  <Square size={14} /> Dừng giám sát
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <ResultsPanel detections={detections} smoothedProb={smoothedProb} faceFound={faceFound} mode="webcam" />
    </div>
  );
}

function WindowMode() {
  const wsRef = useRef(null);
  const [windows, setWindows] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [running, setRunning] = useState(false);
  const [frame, setFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [smoothedProb, setSmoothedProb] = useState(0.5);
  const [faceFound, setFaceFound] = useState(false);
  const [fps, setFps] = useState(0);
  const fpsRef = useRef({ count: 0, last: Date.now() });

  const [scanHeight, setScanHeight] = useState(440);
  const [isExpanded, setIsExpanded] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [warningMsg, setWarningMsg] = useState("");

  const getTargetWin = (id = selectedId) => {
    return windows.find(w => String(w.id ?? w) === String(id)) || null;
  };

  const fetchWindows = async () => {
    try {
      const r = await fetch(`${API}/api/windows`);
      const d = await r.json();
      const list = d.windows || [];
      setWindows(list);
            setSelectedId(prev => {
        if (!list.length) return "";
        const exists = list.some(w => String(w.id ?? w) === String(prev));
        if (exists) return prev;
        return String(list[0].id ?? list[0]);
      });
    } catch {}
  };

  useEffect(() => { fetchWindows(); }, []);

    const handleSelectWindow = (newId) => {
    setSelectedId(newId);
    setErrorMsg("");
    setWarningMsg("");

        if (running && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const target = getTargetWin(newId);
      const winId = target?.id ?? (isNaN(Number(newId)) ? null : Number(newId));
      const winTitle = target?.title ?? newId;
      wsRef.current.send(JSON.stringify({
        action: "switch_window",
        window_id: winId,
        window_title: winTitle
      }));
    }
  };

    const activateWindow = async () => {
    const target = getTargetWin();
    if (!target && !selectedId) return;
    try {
      await fetch(`${API}/api/windows/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          window_id: target?.id ?? null,
          window_title: target?.title ?? selectedId
        })
      });
    } catch (e) {
      console.error("Activate window error:", e);
    }
  };

  const start = useCallback(() => {
    const target = getTargetWin();
    if (!target && !selectedId) return;
    setErrorMsg("");
    setWarningMsg("");

    const ws = new WebSocket(`${API.replace("http", "ws")}/ws/window`);
    const winId = target?.id ?? (isNaN(Number(selectedId)) ? null : Number(selectedId));
    const winTitle = target?.title ?? selectedId;

    ws.onopen = () => {
      ws.send(JSON.stringify({
        window_id: winId,
        window_title: winTitle
      }));
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.error) {
        setErrorMsg(data.error);
        stop();
        return;
      }
      if (data.warning) {
        setWarningMsg(data.warning);
        return;
      } else {
        setWarningMsg("");
      }
      if (data.frame) {
        setFrame(`data:image/jpeg;base64,${data.frame}`);
      }
      setDetections(data.detections || []);
      setSmoothedProb(data.smoothed_prob ?? 0.5);
      setFaceFound(data.face_found ?? false);
      fpsRef.current.count++;
      const now = Date.now();
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count);
        fpsRef.current = { count: 0, last: now };
      }
    };
    ws.onerror = () => { setErrorMsg("Lỗi kết nối WebSocket"); stop(); };
    ws.onclose = () => { setRunning(false); setFrame(null); };
    wsRef.current = ws;
    setRunning(true);
  }, [selectedId, windows]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    setRunning(false); setFrame(null); setFaceFound(false); setDetections([]); setFps(0);
    setWarningMsg("");
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  const activeTitle = getTargetWin()?.title || (selectedId ? String(selectedId) : "Chưa chọn cửa sổ");

  return (
    <div className="tab-content" style={{ ...styles.grid, gridTemplateColumns: isExpanded ? "1fr" : "1fr 400px" }}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Quét Cửa sổ Ứng dụng</span>
          <span style={styles.cardSubtitle}>
            {running ? `Đang quét · ${fps} FPS` : "Sẵn sàng"}
          </span>
        </div>
        <div style={styles.cardBody}>
          <div style={{ display: "flex", gap: "10px", alignItems: "stretch" }}>
            <div style={{ ...styles.videoFrame, height: scanHeight, flex: 1 }}>
              {!frame && <div className="scan-line" />}
              {frame ? (
                <>
                  <img src={frame} alt="Window Capture" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                  <div style={{ position: "absolute", top: 10, left: 10, right: 10, display: "flex", justifyContent: "space-between", pointerEvents: "none" }}>
                    <div style={styles.hudBadge}>
                      Mục tiêu: {activeTitle.length > 32 ? activeTitle.substring(0, 32) + "..." : activeTitle}
                    </div>
                    <div style={styles.hudBadge}>
                      Tốc độ: {fps} FPS
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: "center", color: "#9ca3af" }}>
                  <div style={{ width: 52, height: 52, borderRadius: "50%", backgroundColor: "rgba(0, 59, 113, 0.25)", border: "1px solid rgba(0, 91, 170, 0.4)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 12px" }}>
                    <Monitor size={24} color="#38bdf8" />
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#f3f4f6" }}>
                    {warningMsg || (running ? "Đang kết nối tới cửa sổ..." : "Chưa quét cửa sổ nào")}
                  </div>
                  <div style={{ fontSize: 12, marginTop: 4, color: "#9ca3af" }}>
                    {running ? "Đang nhận luồng khung hình từ cửa sổ mục tiêu" : "Chọn ứng dụng (VLC, Chrome, Zoom) và bấm Bắt đầu quét"}
                  </div>
                </div>
              )}
            </div>

            <div style={styles.sliderContainer}>
              <span style={{ fontSize: 10, fontWeight: 700, color: theme.textMuted }}>H</span>
              <input
                type="range"
                min="250"
                max="800"
                step="10"
                value={scanHeight}
                onChange={e => setScanHeight(Number(e.target.value))}
                orient="vertical"
                style={{
                  WebkitAppearance: "slider-vertical",
                  appearance: "slider-vertical",
                  width: 8,
                  flex: 1,
                  margin: "10px 0",
                  cursor: "pointer",
                }}
              />
              <span style={{ fontSize: 10, fontWeight: 600, color: theme.textMuted }}>{scanHeight}</span>
            </div>
          </div>

          <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div style={{ flex: 1 }}>
                <select
                  value={selectedId}
                  onChange={e => handleSelectWindow(e.target.value)}
                  style={{ ...styles.select, width: "100%", cursor: "pointer" }}
                  title={running ? "Chọn để đổi cửa sổ quét ngay lập tức" : "Chọn cửa sổ để quét"}
                >
                  {windows.length === 0 ? (
                    <option value="">Không tìm thấy cửa sổ nào</option>
                  ) : (
                    windows.map(w => {
                      const id = String(w.id ?? w);
                      const title = w.title ?? w;
                      const prefix = w.is_minimized ? "[Thu nhỏ] " : "";
                      return <option key={id} value={id}>{prefix}{title}</option>;
                    })
                  )}
                </select>
              </div>

              <button
                style={styles.btnOutline}
                onClick={activateWindow}
                title="Hiển thị và đưa cửa sổ đã chọn lên phía trước màn hình"
              >
                <ExternalLink size={14} /> Mở cửa sổ
              </button>

              <button style={styles.btnOutline} onClick={fetchWindows} title="Làm mới danh sách cửa sổ">
                <RefreshCw size={14} /> Tải lại
              </button>

              <button style={styles.btnOutline} onClick={() => setIsExpanded(!isExpanded)}>
                {isExpanded ? <Minimize size={14} /> : <Maximize size={14} />}
              </button>

              {!running ? (
                <button style={styles.btnPrimary} onClick={start} disabled={!selectedId}>
                  <Play size={14} /> Bắt đầu quét
                </button>
              ) : (
                <button style={styles.btnDanger} onClick={stop}>
                  <Square size={14} /> Dừng quét
                </button>
              )}
            </div>

            {errorMsg && (
              <div style={{ padding: "8px 12px", backgroundColor: theme.alertBg, border: `1px solid ${theme.alertBorder}`, color: theme.alertRed, fontSize: 12, borderRadius: "5px" }}>
                Lỗi: {errorMsg}
              </div>
            )}
            {warningMsg && !errorMsg && (
              <div style={{ padding: "8px 12px", backgroundColor: "#fffbeb", border: "1px solid #fef3c7", color: "#b45309", fontSize: 12, borderRadius: "5px" }}>
                ℹ️ {warningMsg}
              </div>
            )}
          </div>
        </div>
      </div>

      <ResultsPanel detections={detections} smoothedProb={smoothedProb} faceFound={faceFound} mode="window" />
    </div>
  );
}

function VideoMode() {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    return () => { if (videoUrl) URL.revokeObjectURL(videoUrl); };
  }, [videoUrl]);

  const analyze = async (f) => {
    setAnalyzing(true); setResult(null); setError(null); setProgress(0);
    const interval = setInterval(() => setProgress(p => Math.min(p + 5, 92)), 180);
    try {
      const fd = new FormData(); fd.append("file", f);
      const r = await fetch(`${API}/api/analyze-video`, { method: "POST", body: fd });
      const d = await r.json();
      if (d.error) setError(d.error); else setResult(d);
    } catch (e) { setError("Không thể kết nối đến máy chủ backend."); }
    clearInterval(interval); setProgress(100); setAnalyzing(false);
  };

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(f));
  };

  const onDrop = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); };

  return (
    <div className="tab-content" style={{ ...styles.grid, gridTemplateColumns: "1fr 400px" }}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Phân tích Tệp tin Video Ngoại tuyến</span>
          <span style={styles.cardSubtitle}>
            {analyzing ? `Đang xử lý (${progress}%)` : "Sẵn sàng"}
          </span>
        </div>
        <div style={styles.cardBody}>
                    <div
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            style={{
              border: `2px dashed ${dragging ? theme.huitNavy : "#9ca3af"}`,
              backgroundColor: dragging ? "#eff6ff" : "rgba(249, 250, 251, 0.7)",
              borderRadius: "6px",
              padding: "2.5rem 1rem",
              textAlign: "center",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            <input ref={inputRef} type="file" accept="video/*" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
            <Upload size={34} color={theme.huitNavy} style={{ margin: "0 auto 10px" }} />
            <div style={{ fontSize: 14, fontWeight: 600, color: theme.textDark }}>
              Kéo thả tệp video vào đây hoặc nhấn để duyệt tệp
            </div>
            <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 4 }}>
              Hỗ trợ định dạng MP4, AVI, MOV, MKV
            </div>
          </div>

                    {videoUrl && (
            <div style={{ marginTop: 14, backgroundColor: "#000", borderRadius: "6px", overflow: "hidden", border: `1px solid ${theme.border}` }}>
              <video src={videoUrl} controls autoPlay muted style={{ width: "100%", maxHeight: 360, display: "block" }} />
            </div>
          )}

                    {file && (
            <div style={{ marginTop: 14, padding: "12px 16px", backgroundColor: "rgba(243, 244, 246, 0.85)", border: `1px solid ${theme.border}`, borderRadius: "6px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div>
                <span style={{ fontSize: 13, fontWeight: 600, color: theme.textDark }}>{file.name}</span>
                <span style={{ fontSize: 12, color: theme.textMuted, marginLeft: 8 }}>({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
              </div>
              <button style={styles.btnPrimary} onClick={() => analyze(file)} disabled={analyzing}>
                {analyzing ? <RefreshCw size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Play size={14} />}
                {analyzing ? "Đang xử lý..." : "Khởi chạy phân tích"}
              </button>
            </div>
          )}

                    {analyzing && (
            <div style={{ marginTop: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: theme.textMuted, marginBottom: 5 }}>
                <span>Đang trích xuất và phân tích khung hình tuần tự...</span>
                <span style={{ fontWeight: 600, color: theme.huitNavy }}>{progress}%</span>
              </div>
              <div style={{ height: 6, backgroundColor: "#e5e7eb", borderRadius: "3px", overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${progress}%`, backgroundColor: theme.huitNavy, transition: "width 0.2s linear" }} />
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: 14, padding: "10px 14px", backgroundColor: theme.alertBg, border: `1px solid ${theme.alertBorder}`, color: theme.alertRed, fontSize: 12, borderRadius: "5px" }}>
              Lỗi: {error}
            </div>
          )}
        </div>
      </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.cardTitle}>Báo cáo Giám định Tệp tin</span>
          </div>
          <div style={styles.cardBody}>
            {result ? (
              <div>
                <div
                  style={{
                    padding: "16px",
                    borderRadius: "6px",
                    backgroundColor: result.verdict === "DEEPFAKE" ? theme.alertBg : theme.successBg,
                    border: `1px solid ${result.verdict === "DEEPFAKE" ? theme.alertBorder : theme.successBorder}`,
                    textAlign: "center",
                    marginBottom: 14,
                    animation: "fadeIn 0.25s ease",
                  }}
                >
                  <div style={{ fontSize: 11, fontWeight: 700, color: result.verdict === "DEEPFAKE" ? theme.alertRed : theme.successGreen, letterSpacing: "0.5px" }}>
                    KẾT LUẬN GIÁM ĐỊNH
                  </div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: result.verdict === "DEEPFAKE" ? theme.alertRed : theme.successGreen, margin: "6px 0" }}>
                    {result.verdict === "DEEPFAKE" ? "DEEPFAKE (GIẢ MẠO)" : "GENUINE (XÁC THỰC)"}
                  </div>
                  <div style={{ fontSize: 12, color: theme.textMuted }}>
                    Độ tin cậy: <strong>{result.confidence}%</strong> · Xác suất trung bình: <strong>{(result.average_probability * 100).toFixed(1)}%</strong>
                  </div>
                </div>

                <MetricGauge prob={result.average_probability} label="Xác suất Deepfake trung bình video" />

                <table style={{ ...styles.table, marginTop: 14 }}>
                  <tbody>
                    <tr style={styles.tableRow}>
                      <td style={styles.tableKey}>Khung hình phân tích</td>
                      <td style={styles.tableVal}>{result.total_frames_analyzed} frames</td>
                    </tr>
                    <tr style={styles.tableRow}>
                      <td style={styles.tableKey}>Xác suất Deepfake trung bình</td>
                      <td style={styles.tableVal}>{(result.average_probability * 100).toFixed(2)}%</td>
                    </tr>
                    <tr style={styles.tableRow}>
                      <td style={styles.tableKey}>Ngưỡng quyết định EER</td>
                      <td style={styles.tableVal}>T = 0.78 (Calibrated)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: "2.5rem 1rem", color: theme.textLight }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: theme.textMuted }}>Chưa có kết quả giám định</div>
                <div style={{ fontSize: 12, marginTop: 4 }}>Tải tệp video lên và bấm Khởi chạy phân tích để xem kết quả</div>
              </div>
            )}
          </div>
        </div>

        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.cardTitle}>Phương pháp luận (Methodology)</span>
          </div>
          <div style={styles.cardBody}>
            <div style={{ fontSize: 12, color: theme.textMuted, lineHeight: 1.7 }}>
              Theo đề cương nghiên cứu <strong>CNTT-KLCN174</strong>, mô hình <strong>DeepfakeFusionModel</strong> khai thác đồng thời 2 nguồn đặc trưng độc lập:
            </div>
            <ul style={{ fontSize: 12, color: theme.textMuted, marginTop: 8, paddingLeft: 18, lineHeight: 1.8 }}>
              <li><strong>Nhánh không gian (Swin Transformer):</strong> Trích xuất đặc trưng thị giác vi mô (texture, blending artifact).</li>
              <li><strong>Nhánh sinh lý (rPPG):</strong> Ước lượng biến thiên chu kỳ tim mạch từ biến đổi màu sắc mao mạch da mặt qua các khung hình liên tiếp.</li>
              <li><strong>Hợp nhất Cross-Attention (CAN):</strong> Tự động học mối liên hệ Query-Key-Value giữa 2 nguồn đặc trưng để phân loại nhị phân Real/Fake.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("webcam");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "error" }));
  }, []);

  const tabs = [
    { id: "webcam", icon: Camera, label: "GIÁM SÁT WEBCAM TRỰC TIẾP" },
    { id: "window", icon: Monitor, label: "QUÉT CỬA SỔ ỨNG DỤNG" },
    { id: "video", icon: Film, label: "PHÂN TÍCH TỆP TIN VIDEO" },
  ];

  return (
    <div style={styles.appWrapper}>
            <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.brandGroup}>
            <img src="/huit-logo.png" alt="Logo HUIT" style={styles.logoImg} />
            <div>
              <div style={styles.instTitle}>TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH · HUIT</div>
              <div style={styles.deptTitle}>KHOA CÔNG NGHỆ THÔNG TIN</div>
              <div style={styles.headerTitle}>HỆ THỐNG PHÁT HIỆN DEEPFAKE KẾT HỢP ĐẶC TRƯNG KHÔNG GIAN VÀ rPPG (CNTT-KLCN174)</div>
            </div>
          </div>

          <div style={styles.statusGroup}>
            {health && (
              <div style={styles.statusPill(health.status === "ok")}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", backgroundColor: health.status === "ok" ? "#059669" : "#dc2626" }} />
                <span>Backend: {health.status === "ok" ? `Sẵn sàng (${health.device?.toUpperCase()})` : "Ngoại tuyến"}</span>
              </div>
            )}
            {health?.model_loaded && (
              <div style={styles.modelPill}>
                <span>Trọng số: Calibrated (T=0.78)</span>
              </div>
            )}
          </div>
        </div>
      </header>

            <main style={styles.main}>
                <div style={styles.navBar}>
          {tabs.map(({ id, icon: Icon, label }) => (
            <button key={id} style={styles.navBtn(tab === id)} onClick={() => setTab(id)}>
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </div>

        {tab === "webcam" && <WebcamMode />}
        {tab === "window" && <WindowMode />}
        {tab === "video" && <VideoMode />}
      </main>

            <footer style={styles.footer}>
        <div style={styles.footerInner}>
          <div>
            <div style={styles.footerColTitle}>ĐƠN VỊ ĐÀO TẠO & NGHIÊN CỨU</div>
            <div style={styles.footerText}>
              <strong>Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)</strong><br />
              <strong>Khoa Công nghệ Thông tin</strong><br />
              Khóa luận Cử nhân ngành CNTT (Năm học 2025 – 2026)<br />
              Địa chỉ: 140 Lê Trọng Tấn, Phường Tây Thạnh, Quận Tân Phú, TP. Hồ Chí Minh<br />
              Cổng thông tin: <a href="https://huit.edu.vn" target="_blank" rel="noreferrer" style={{ color: theme.huitNavy, textDecoration: "none", fontWeight: 600 }}>huit.edu.vn</a>
            </div>
          </div>

          <div>
            <div style={styles.footerColTitle}>ĐỀ TÀI KHÓA LUẬN (MÃ: CNTT-KLCN174)</div>
            <div style={styles.footerText}>
              <strong>Tên đề tài:</strong> Phương pháp lai kết hợp đặc trưng không gian và tín hiệu sinh lý để phát hiện Deepfake sử dụng mô hình học sâu và rPPG<br />
              <strong>Mô hình đề xuất:</strong> DeepfakeFusionModel (Swin Transformer + rPPG + Cross-Attention Network)<br />
              <strong>Dữ liệu thực nghiệm:</strong> FaceForensics++ C23 & Celeb-DF v2<br />
              <strong>Độ đo đánh giá:</strong> Accuracy, AUC, EER (Ngưỡng hiệu chuẩn T=0.78), FAR/FRR
            </div>
          </div>

          <div>
            <div style={styles.footerColTitle}>THÀNH VIÊN THỰC HIỆN ĐỀ TÀI</div>
            <div style={styles.footerText}>
              <strong>GVHD:</strong> ThS. Phùng Thế Bảo (baopt@huit.edu.vn)<br />
              <div style={{ marginTop: 6, borderTop: "1px dashed #e5e7eb", paddingTop: 6 }}>
                <strong>Nhóm sinh viên thực hiện:</strong><br />
                • Nguyễn Thành Tài — MSSV: 2001230773 (Lớp: 14DHTH05)<br />
                • Phan Sĩ Thịnh — MSSV: 2001230933 (Lớp: 14DHTH05)<br />
                • Đoàn Tấn Minh Tân — MSSV: 2001230791 (Lớp: 14DHTH15)
              </div>
            </div>
          </div>
        </div>

        <div style={styles.footerBottomBar}>
          © 2025 – 2026 Khoa Công Nghệ Thông Tin · Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) · Đề tài CNTT-KLCN174.
        </div>
      </footer>
    </div>
  );
}
