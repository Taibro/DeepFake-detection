import { useState, useEffect, useRef, useCallback } from "react";
import { Camera, Monitor, Film, Wifi, WifiOff, RefreshCw, Play, Square, Upload, AlertCircle, CheckCircle, Activity, ChevronDown, Zap, Eye, Maximize, Minimize, SlidersHorizontal } from "lucide-react";

const API = "http://localhost:8000";

const palette = {
  teal50: "#E1F5EE", teal100: "#9FE1CB", teal400: "#1D9E75", teal600: "#0F6E56",
  blue50: "#E6F1FB", blue100: "#B5D4F4", blue400: "#378ADD", blue600: "#185FA5",
  green50: "#EAF3DE", green400: "#639922",
};

const styles = {
  app: { minHeight: "100vh", background: "linear-gradient(135deg, #E6F1FB 0%, #E1F5EE 50%, #EAF3DE 100%)", fontFamily: "'DM Sans', system-ui, sans-serif", padding: "0" },
  header: { background: "rgba(255,255,255,0.85)", backdropFilter: "blur(20px)", borderBottom: "1px solid rgba(29,158,117,0.15)", padding: "0 2rem", position: "sticky", top: 0, zIndex: 100 },
  headerInner: { maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 },
  logo: { display: "flex", alignItems: "center", gap: 10 },
  logoIcon: { width: 36, height: 36, background: "linear-gradient(135deg, #1D9E75, #378ADD)", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center" },
  logoText: { fontSize: 17, fontWeight: 700, background: "linear-gradient(135deg, #0F6E56, #185FA5)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" },
  badge: (ok) => ({ display: "flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600, background: ok ? "#E1F5EE" : "#FCEBEB", color: ok ? "#0F6E56" : "#A32D2D", border: `1px solid ${ok ? "#9FE1CB" : "#F7C1C1"}` }),
  main: { maxWidth: 1100, margin: "0 auto", padding: "2rem" },
  tabs: { display: "flex", gap: 8, marginBottom: "1.5rem", background: "rgba(255,255,255,0.7)", padding: 6, borderRadius: 16, border: "1px solid rgba(29,158,117,0.15)", width: "fit-content" },
  tab: (active) => ({ display: "flex", alignItems: "center", gap: 8, padding: "10px 20px", borderRadius: 12, border: "none", cursor: "pointer", fontSize: 14, fontWeight: active ? 600 : 400, transition: "all 0.2s", background: active ? "linear-gradient(135deg, #1D9E75, #378ADD)" : "transparent", color: active ? "white" : "#5F5E5A" }),
  grid: { display: "grid", gap: "1.5rem", alignItems: "start", transition: "all 0.3s ease" },
  card: { background: "rgba(255,255,255,0.9)", borderRadius: 20, border: "1px solid rgba(29,158,117,0.12)", overflow: "hidden", boxShadow: "0 4px 24px rgba(29,158,117,0.08)", transition: "all 0.3s ease" },
  cardHeader: { padding: "1.25rem 1.5rem", borderBottom: "1px solid rgba(29,158,117,0.1)", display: "flex", alignItems: "center", justifyContent: "space-between" },
  cardTitle: { fontSize: 15, fontWeight: 600, color: "#2c3e50", display: "flex", alignItems: "center", gap: 8 },
  cardBody: { padding: "1.5rem" },
  videoBox: { background: "#0a1628", borderRadius: 12, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", transition: "height 0.2s ease" },
  placeholder: { display: "flex", flexDirection: "column", alignItems: "center", gap: 12, color: "rgba(255,255,255,0.4)" },
  btn: (variant = "primary") => ({
    display: "flex", alignItems: "center", gap: 8, padding: "11px 22px", borderRadius: 12, border: "none", cursor: "pointer", fontSize: 14, fontWeight: 600, transition: "all 0.2s",
    background: variant === "primary" ? "linear-gradient(135deg, #1D9E75, #378ADD)" : variant === "danger" ? "#E24B4A" : variant === "outline" ? "transparent" : "rgba(29,158,117,0.1)",
    color: variant === "outline" ? "#1D9E75" : "white",
    border: variant === "outline" ? "1px solid #1D9E75" : "none",
  }),
  meter: (prob, fake) => ({ width: "100%", height: 8, background: "#E1F5EE", borderRadius: 4, overflow: "hidden", position: "relative" }),
  meterFill: (prob, fake) => ({ height: "100%", width: `${prob * 100}%`, background: fake ? "linear-gradient(90deg, #EF9F27, #E24B4A)" : "linear-gradient(90deg, #1D9E75, #378ADD)", borderRadius: 4, transition: "width 0.4s ease" }),
  statRow: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: "1px solid rgba(29,158,117,0.08)" },
  select: { width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid rgba(29,158,117,0.25)", background: "white", fontSize: 14, color: "#2c3e50", appearance: "none", cursor: "pointer" },
  uploadZone: (drag) => ({ border: `2px dashed ${drag ? "#1D9E75" : "rgba(29,158,117,0.3)"}`, borderRadius: 16, padding: "3rem", display: "flex", flexDirection: "column", alignItems: "center", gap: 12, cursor: "pointer", transition: "all 0.2s", background: drag ? "rgba(29,158,117,0.05)" : "transparent" }),
  pill: (fake) => ({ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 700, background: fake ? "#FCEBEB" : "#E1F5EE", color: fake ? "#A32D2D" : "#0F6E56" }),
  pulse: { width: 8, height: 8, borderRadius: "50%", background: "#1D9E75", animation: "pulse 1.5s infinite" },
};

function ProbabilityMeter({ prob, label }) {
  const isFake = prob > 0.5;
  const displayPct = isFake ? prob * 100 : (1 - prob) * 100;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 13, color: "#5F5E5A" }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color: isFake ? "#A32D2D" : "#0F6E56" }}>{displayPct.toFixed(1)}%</span>
      </div>
      <div style={styles.meter(prob, isFake)}>
        <div style={styles.meterFill(isFake ? prob : 1 - prob, isFake)} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        <span style={{ fontSize: 11, color: "#1D9E75", fontWeight: 600 }}>REAL</span>
        <span style={{ fontSize: 11, color: "#E24B4A", fontWeight: 600 }}>DEEPFAKE</span>
      </div>
    </div>
  );
}

function ResultsPanel({ detections, smoothedProb, faceFound, mode }) {
  const isFake = smoothedProb > 0.5;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem", height: "100%" }}>
      <div style={{ ...styles.card, flex: 1 }}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}><Activity size={16} color="#1D9E75" />Live Analysis</span>
          {faceFound && <div style={styles.pulse} />}
        </div>
        <div style={styles.cardBody}>
          {faceFound ? (
            <>
              <div style={{ textAlign: "center", marginBottom: "1.25rem" }}>
                <div style={{ fontSize: 13, color: "#888", marginBottom: 4 }}>Verdict</div>
                <div style={{ fontSize: 28, fontWeight: 800, color: isFake ? "#A32D2D" : "#0F6E56" }}>
                  {isFake ? "DEEPFAKE" : "GENUINE"}
                </div>
              </div>
              <ProbabilityMeter prob={smoothedProb} label="Detection confidence" />
              <div style={{ marginTop: "1rem" }}>
                {detections.map((d, i) => (
                  <div key={i} style={styles.statRow}>
                    <span style={{ fontSize: 13, color: "#5F5E5A" }}>Face #{i + 1}</span>
                    <span style={styles.pill(d.is_fake)}>{d.is_fake ? "Fake" : "Real"} {d.confidence.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "1.5rem 0", color: "#888" }}>
              <Eye size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
              <div style={{ fontSize: 13 }}>No face detected</div>
              <div style={{ fontSize: 11, marginTop: 4 }}>Position face in frame</div>
            </div>
          )}
        </div>
      </div>

      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}><Zap size={16} color="#378ADD" />How it works</span>
        </div>
        <div style={{ padding: "1rem 1.5rem", display: "flex", flexDirection: "column", gap: 10 }}>
          {[["Swin Transformer", "Visual artifact detection"], ["rPPG Analysis", "Physiological signal check"], ["Cross-Attention Fusion", "Multi-modal confidence"]].map(([name, desc]) => (
            <div key={name} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#1D9E75", marginTop: 5, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#2c3e50" }}>{name}</div>
                <div style={{ fontSize: 11, color: "#888" }}>{desc}</div>
              </div>
            </div>
          ))}
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

  const [scanHeight, setScanHeight] = useState(400);
  const [isExpanded, setIsExpanded] = useState(false);

  const start = useCallback(() => {
    const ws = new WebSocket(`${API.replace("http", "ws")}/ws/webcam`);
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
  }, []);

  const stop = useCallback(() => {
    wsRef.current?.close();
    setRunning(false); setFrame(null); setFaceFound(false); setDetections([]); setFps(0);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  return (
    <div style={{ ...styles.grid, gridTemplateColumns: isExpanded ? "1fr" : "1fr 360px" }}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}><Camera size={16} color="#1D9E75" />Webcam Feed</span>
          {running && <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 11, color: "#1D9E75", fontWeight: 600 }}>{fps} FPS</span>
            <div style={styles.pulse} />
          </div>}
        </div>
        <div style={styles.cardBody}>
          <div style={{ display: "flex", gap: "1rem", alignItems: "stretch" }}>
            {/* Khu vực Video */}
            <div style={{ ...styles.videoBox, height: scanHeight, flex: 1, aspectRatio: "auto" }}>
              {frame ? <img src={frame} alt="live" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> :
                <div style={styles.placeholder}>
                  <Camera size={48} />
                  <div style={{ fontSize: 14 }}>Camera preview will appear here</div>
                </div>}
            </div>

            {/* Thanh slider nằm dọc bên cạnh video */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "space-between", background: "rgba(29,158,117,0.05)", padding: "15px 10px", borderRadius: "12px" }}>
              <SlidersHorizontal size={16} color="#1D9E75" style={{ transform: "rotate(90deg)" }} />
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
                  margin: "15px 0",
                  cursor: "pointer", 
                  accentColor: "#1D9E75" 
                }} 
              />
              <span style={{ fontSize: 12, color: "#888", fontWeight: 600 }}>{scanHeight}</span>
            </div>
          </div>

          <div style={{ marginTop: "1.25rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <button style={{ ...styles.btn("outline"), padding: "6px 12px", fontSize: 12 }} onClick={() => setIsExpanded(!isExpanded)}>
              {isExpanded ? <><Minimize size={14} /> Thu gọn</> : <><Maximize size={14} /> Mở rộng</>}
            </button>
            <div style={{ display: "flex", gap: 10 }}>
              {!running ? <button style={styles.btn("primary")} onClick={start}><Play size={16} />Start Camera</button> :
                <button style={styles.btn("danger")} onClick={stop}><Square size={14} />Stop</button>}
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
  const [selected, setSelected] = useState("");
  const [running, setRunning] = useState(false);
  const [frame, setFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [smoothedProb, setSmoothedProb] = useState(0.5);
  const [faceFound, setFaceFound] = useState(false);
  const [fps, setFps] = useState(0);
  const fpsRef = useRef({ count: 0, last: Date.now() });

  const [scanHeight, setScanHeight] = useState(400);
  const [isExpanded, setIsExpanded] = useState(false);

  const fetchWindows = async () => {
    try { const r = await fetch(`${API}/api/windows`); const d = await r.json(); setWindows(d.windows || []); if (d.windows?.length) setSelected(d.windows[0]); } catch {}
  };

  useEffect(() => { fetchWindows(); }, []);

  const start = useCallback(() => {
    if (!selected) return;
    const ws = new WebSocket(`${API.replace("http", "ws")}/ws/window`);
    ws.onopen = () => ws.send(JSON.stringify({ window_title: selected }));
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.error) { stop(); return; }
      setFrame(`data:image/jpeg;base64,${data.frame}`);
      setDetections(data.detections || []);
      setSmoothedProb(data.smoothed_prob ?? 0.5);
      setFaceFound(data.face_found ?? false);
      fpsRef.current.count++;
      const now = Date.now();
      if (now - fpsRef.current.last >= 1000) { setFps(fpsRef.current.count); fpsRef.current = { count: 0, last: now }; }
    };
    ws.onerror = () => stop();
    ws.onclose = () => { setRunning(false); setFrame(null); };
    wsRef.current = ws;
    setRunning(true);
  }, [selected]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    setRunning(false); setFrame(null); setFaceFound(false); setDetections([]); setFps(0);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  return (
    <div style={{ ...styles.grid, gridTemplateColumns: isExpanded ? "1fr" : "1fr 360px" }}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}><Monitor size={16} color="#378ADD" />Window Capture</span>
          {running && <span style={{ fontSize: 11, color: "#1D9E75", fontWeight: 600 }}>{fps} FPS</span>}
        </div>
        <div style={styles.cardBody}>
          <div style={{ display: "flex", gap: "1rem", alignItems: "stretch" }}>
            {/* Khu vực Video */}
            <div style={{ ...styles.videoBox, height: scanHeight, flex: 1, aspectRatio: "auto" }}>
              {frame ? <img src={frame} alt="window" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> :
                <div style={styles.placeholder}><Monitor size={48} /><div style={{ fontSize: 14 }}>Select a window and start scanning</div></div>}
            </div>

            {/* Thanh slider nằm dọc bên cạnh video */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "space-between", background: "rgba(29,158,117,0.05)", padding: "15px 10px", borderRadius: "12px" }}>
              <SlidersHorizontal size={16} color="#1D9E75" style={{ transform: "rotate(90deg)" }} />
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
                  margin: "15px 0",
                  cursor: "pointer", 
                  accentColor: "#1D9E75" 
                }} 
              />
              <span style={{ fontSize: 12, color: "#888", fontWeight: 600 }}>{scanHeight}</span>
            </div>
          </div>

          <div style={{ marginTop: "1.25rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: "1rem" }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#5F5E5A", marginBottom: 8 }}>Target Window</div>
                <div style={{ position: "relative" }}>
                  <select value={selected} onChange={e => setSelected(e.target.value)} style={styles.select} disabled={running}>
                    {windows.length === 0 ? <option>No windows found</option> : windows.map(w => <option key={w}>{w}</option>)}
                  </select>
                  <ChevronDown size={14} style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "#888", pointerEvents: "none" }} />
                </div>
              </div>
              <button style={{ ...styles.btn("outline"), padding: "0 12px", fontSize: 12, height: "39px" }} onClick={() => setIsExpanded(!isExpanded)}>
                {isExpanded ? <><Minimize size={14} /> Thu gọn</> : <><Maximize size={14} /> Mở rộng</>}
              </button>
            </div>

            <div style={{ display: "flex", gap: 10 }}>
              {!running ? <button style={styles.btn("primary")} onClick={start} disabled={!selected}><Play size={16} />Start Scan</button> :
                <button style={styles.btn("danger")} onClick={stop}><Square size={14} />Stop</button>}
              <button style={styles.btn("outline")} onClick={fetchWindows}><RefreshCw size={14} />Refresh</button>
            </div>
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
  const inputRef = useRef(null);

  const analyze = async (f) => {
    setAnalyzing(true); setResult(null); setError(null); setProgress(0);
    const interval = setInterval(() => setProgress(p => Math.min(p + 2, 90)), 300);
    try {
      const fd = new FormData(); fd.append("file", f);
      const r = await fetch(`${API}/api/analyze-video`, { method: "POST", body: fd });
      const d = await r.json();
      if (d.error) setError(d.error); else setResult(d);
    } catch (e) { setError("Failed to connect to backend."); }
    clearInterval(interval); setProgress(100); setAnalyzing(false);
  };

  const handleFile = (f) => { if (!f) return; setFile(f); setResult(null); setError(null); };

  const onDrop = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); };

  return (
    <div style={{ ...styles.grid, gridTemplateColumns: "1fr 360px" }}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}><Film size={16} color="#639922" />Video Analysis</span>
        </div>
        <div style={styles.cardBody}>
          <div style={styles.uploadZone(dragging)} onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <input ref={inputRef} type="file" accept="video/*" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
            <div style={{ width: 56, height: 56, borderRadius: 16, background: "linear-gradient(135deg, #E1F5EE, #E6F1FB)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Upload size={24} color="#1D9E75" />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#2c3e50", textAlign: "center" }}>Drop video file here</div>
              <div style={{ fontSize: 13, color: "#888", textAlign: "center", marginTop: 4 }}>or click to browse — MP4, MOV, AVI</div>
            </div>
          </div>

          {file && (
            <div style={{ marginTop: "1rem", padding: "12px 16px", background: "#E6F1FB", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Film size={16} color="#185FA5" />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#2c3e50" }}>{file.name}</div>
                  <div style={{ fontSize: 11, color: "#888" }}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                </div>
              </div>
              <button style={styles.btn("primary")} onClick={() => analyze(file)} disabled={analyzing}>
                {analyzing ? <RefreshCw size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Play size={14} />}
                {analyzing ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          )}

          {analyzing && (
            <div style={{ marginTop: "1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: "#5F5E5A" }}>Processing frames...</span>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#1D9E75" }}>{progress}%</span>
              </div>
              <div style={{ height: 6, background: "#E1F5EE", borderRadius: 3, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${progress}%`, background: "linear-gradient(90deg, #1D9E75, #378ADD)", borderRadius: 3, transition: "width 0.3s" }} />
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: "1rem", padding: "12px 16px", background: "#FCEBEB", borderRadius: 12, display: "flex", gap: 10, alignItems: "center" }}>
              <AlertCircle size={16} color="#A32D2D" /><span style={{ fontSize: 13, color: "#A32D2D" }}>{error}</span>
            </div>
          )}

          {result && (
            <div style={{ marginTop: "1.25rem" }}>
              <div style={{ padding: "1.25rem", borderRadius: 16, background: result.verdict === "DEEPFAKE" ? "#FCEBEB" : "#E1F5EE", border: `1.5px solid ${result.verdict === "DEEPFAKE" ? "#F7C1C1" : "#9FE1CB"}`, marginBottom: "1rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  {result.verdict === "DEEPFAKE" ? <AlertCircle size={28} color="#A32D2D" /> : <CheckCircle size={28} color="#0F6E56" />}
                  <div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: result.verdict === "DEEPFAKE" ? "#A32D2D" : "#0F6E56" }}>{result.verdict}</div>
                    <div style={{ fontSize: 13, color: "#888" }}>Confidence: {result.confidence}%</div>
                  </div>
                </div>
              </div>
              <ProbabilityMeter prob={result.average_probability} label="Average deepfake probability" />
              <div style={{ marginTop: "1rem" }}>
                {[["Frames analyzed", result.total_frames_analyzed], ["Avg probability", `${(result.average_probability * 100).toFixed(1)}%`]].map(([k, v]) => (
                  <div key={k} style={styles.statRow}><span style={{ fontSize: 13, color: "#5F5E5A" }}>{k}</span><span style={{ fontSize: 13, fontWeight: 600, color: "#2c3e50" }}>{v}</span></div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <div style={styles.card}>
          <div style={styles.cardHeader}><span style={styles.cardTitle}><Activity size={16} color="#639922" />Analysis Info</span></div>
          <div style={{ padding: "1rem 1.5rem" }}>
            <div style={{ fontSize: 13, color: "#5F5E5A", lineHeight: 1.8 }}>
              The video is sampled at up to <strong>120 frames</strong>, evenly spaced. Each frame is analyzed with the Swin Transformer + rPPG fusion model. Results are aggregated for a final verdict.
            </div>
            <div style={{ marginTop: "1rem", display: "flex", flexDirection: "column", gap: 8 }}>
              {[["Max frames sampled", "120"], ["Face detection", "MediaPipe"], ["Model", "Swin + rPPG Fusion"]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 12, color: "#888" }}>{k}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "#2c3e50" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div style={styles.card}>
          <div style={styles.cardHeader}><span style={styles.cardTitle}><Zap size={16} color="#378ADD" />Tips</span></div>
          <div style={{ padding: "1rem 1.5rem", display: "flex", flexDirection: "column", gap: 10 }}>
            {["Videos with clear, well-lit faces produce most accurate results.", "Longer videos give more data points for reliable analysis.", "Supports MP4, MOV, AVI, and most common formats."].map((t, i) => (
              <div key={i} style={{ display: "flex", gap: 10, fontSize: 12, color: "#5F5E5A", lineHeight: 1.6 }}>
                <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#378ADD", marginTop: 5, flexShrink: 0 }} />{t}
              </div>
            ))}
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
    fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => setHealth({ status: "error" }));
  }, []);

  const tabs = [
    { id: "webcam", icon: Camera, label: "Webcam" },
    { id: "window", icon: Monitor, label: "Window Capture" },
    { id: "video", icon: Film, label: "Video File" },
  ];

  return (
    <div style={styles.app}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        button:not(:disabled):hover { filter: brightness(1.08); }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        select:focus, input:focus { outline: 2px solid rgba(29,158,117,0.4); outline-offset: 1px; }
      `}</style>

      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}><Eye size={20} color="white" /></div>
            <span style={styles.logoText}>HUIT Deepfake Scanner</span>
            <span style={{ fontSize: 11, padding: "2px 8px", background: "#E1F5EE", color: "#0F6E56", borderRadius: 6, fontWeight: 600 }}>v2.0</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {health && (
              <div style={styles.badge(health.status === "ok")}>
                {health.status === "ok" ? <Wifi size={12} /> : <WifiOff size={12} />}
                {health.status === "ok" ? `Backend ready · ${health.device?.toUpperCase()}` : "Backend offline"}
              </div>
            )}
            {health?.model_loaded === false && (
              <div style={{ ...styles.badge(false), background: "#FAEEDA", color: "#854F0B", border: "1px solid #FAC775" }}>
                <AlertCircle size={12} />Random weights
              </div>
            )}
          </div>
        </div>
      </header>

      <main style={styles.main}>
        <div style={{ marginBottom: "1.5rem" }}>
          <h1 style={{ fontSize: 26, fontWeight: 800, background: "linear-gradient(135deg, #0F6E56, #185FA5)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 6 }}>
            Real-time Deepfake Detection
          </h1>
          <p style={{ fontSize: 14, color: "#5F5E5A" }}>
            Powered by Swin Transformer + rPPG Fusion — choose a detection mode below
          </p>
        </div>

        <div style={styles.tabs}>
          {tabs.map(({ id, icon: Icon, label }) => (
            <button key={id} style={styles.tab(tab === id)} onClick={() => setTab(id)}>
              <Icon size={15} />{label}
            </button>
          ))}
        </div>

        {tab === "webcam" && <WebcamMode />}
        {tab === "window" && <WindowMode />}
        {tab === "video" && <VideoMode />}
      </main>
    </div>
  );
}