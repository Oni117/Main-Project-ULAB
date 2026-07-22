import { useState, useCallback, useEffect } from 'react'
import UploadZone from './components/UploadZone'
import ImagePreview from './components/ImagePreview'
import ResultCard from './components/ResultCard'
import GradCAMPanel from './components/GradCAMPanel'
import { analyzeImage, fetchGradCAM, checkHealth } from './api'
import s from './App.module.css'

const MODEL_NAME = 'efficientnet_b1_cbam'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)

  // Grad-CAM data is an object:
  // { original, heatmap, overlay, pred_class, confidence }
  const [camData, setCamData] = useState(null)

  const [loading, setLoading] = useState(false)
  const [camLoading, setCamLoading] = useState(false)

  const [error, setError] = useState(null)
  const [camError, setCamError] = useState(null)

  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'))
  }, [])

  const handleFile = useCallback(
    (f) => {
      if (preview) URL.revokeObjectURL(preview)

      setFile(f)
      setPreview(URL.createObjectURL(f))

      setResult(null)
      setCamData(null)
      setError(null)
      setCamError(null)
    },
    [preview]
  )

  const handleReset = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview)

    setFile(null)
    setPreview(null)

    setResult(null)
    setCamData(null)
    setError(null)
    setCamError(null)
  }, [preview])

  async function handleAnalyze() {
    if (!file || loading) return

    setLoading(true)
    setError(null)
    setResult(null)

    setCamData(null)
    setCamError(null)

    try {
      // 1. Prediction JSON
      const data = await analyzeImage(file, MODEL_NAME)
      setResult(data)

      // 2. Grad-CAM JSON object
      setCamLoading(true)

      try {
        const cam = await fetchGradCAM(file, MODEL_NAME)
        setCamData(cam)
      } catch (ce) {
        setCamError(ce.message || 'Grad-CAM failed')
      } finally {
        setCamLoading(false)
      }
    } catch (e) {
      setError(e.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.logo}>
          <span className={s.logoMark} />
          TongueCheck
        </div>

        <div className={s.headerCenter}>
          <span className={s.modelPill}>EfficientNet-B1 + CBAM</span>
          <span className={s.headerSubtitle}>
            Tongue-based diabetes screening
          </span>
        </div>

        <div className={s.headerRight}>
          <StatusDot status={apiStatus} />
        </div>
      </header>

      <main className={s.main}>
        <section className={s.left}>
          <p className={s.sectionLabel}>Upload image</p>

          <UploadZone onFile={handleFile} disabled={loading} />

          {file && preview && (
            <div className="anim-up" style={{ marginTop: 16 }}>
              <ImagePreview
                file={file}
                src={preview}
                onRemove={handleReset}
                disabled={loading}
              />
            </div>
          )}

          <div className={s.modelStrip}>
            <div className={s.modelStripLeft}>
              <span className={s.modelStripDot} />
              <div>
                <p className={s.modelStripName}>EfficientNet-B1 + CBAM</p>
                <p className={s.modelStripMeta}>
                  240 × 240 · ImageNet normalisation · 2-class softmax
                </p>
              </div>
            </div>
            <span className={s.modelStripBadge}>Active</span>
          </div>

          <button
            className={s.analyzeBtn}
            onClick={handleAnalyze}
            disabled={!file || loading || apiStatus !== 'ok'}
          >
            {loading ? (
              <>
                <span className={s.spinner} />
                Analyzing…
              </>
            ) : (
              'Analyze image'
            )}
          </button>

          {apiStatus === 'error' && (
            <p className={s.apiWarn}>
              Cannot reach backend. Run:{' '}
              <code>python -m uvicorn main:app --reload --port 8000</code>
            </p>
          )}
        </section>

        <section className={s.right}>
          <p className={s.sectionLabel}>Result</p>

          {!result && !error && !loading && (
            <div className={s.empty}>
              <div className={s.emptyIcon}>
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinecap="round"
                >
                  <circle cx="11" cy="11" r="8" />
                  <path d="M21 21l-4.3-4.3" />
                </svg>
              </div>
              <p>
                Upload a tongue image and click <strong>Analyze image</strong>{' '}
                to see the prediction.
              </p>
            </div>
          )}

          {loading && (
            <div className={s.loadingBox}>
              <span className={s.loadingRing} />
              <p>Running inference…</p>
            </div>
          )}

          {error && (
            <div className={s.errorBox}>
              <strong>Error</strong>
              <p>{error}</p>
            </div>
          )}

          {result && <ResultCard result={result} />}

          {(result || camLoading || camError) && (
            <div className="anim-up delay-1">
              <GradCAMPanel
                data={camData}
                result={result}
                loading={camLoading}
                error={camError}
              />
            </div>
          )}

          {!result && !loading && (
            <div className={s.infoCard}>
              <h3 className={s.infoTitle}>About the model</h3>
              <p>
                EfficientNet-B1 enhanced with a CBAM (Convolutional Block
                Attention Module) trained on tongue images at 240 × 240 px. CBAM
                applies both channel-wise and spatial attention, helping the
                model focus on the most discriminative tongue surface features.
                Grad-CAM highlights the exact regions that drove each
                prediction.
              </p>

              <div className={s.infoGrid}>
                <Stat label="Architecture" value="EfficientNet-B1" />
                <Stat label="Attention" value="CBAM" />
                <Stat label="Input size" value="240 × 240" />
                <Stat label="Classes" value="Diabetic · Non-Diabetic" />
                <Stat label="Output" value="Softmax" />
                <Stat label="Explainability" value="Grad-CAM" />
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className={s.footer}>
        For academic and research use only · Not a certified medical device
      </footer>
    </div>
  )
}

function StatusDot({ status }) {
  const color = {
    checking: '#aaa',
    ok: '#1B6B49',
    error: '#A32B1D',
  }[status]

  const label = {
    checking: 'Connecting…',
    ok: 'API online',
    error: 'API offline',
  }[status]

  return (
    <span
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 12,
        color: 'var(--muted)',
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: color,
          display: 'inline-block',
        }}
      />
      {label}
    </span>
  )
}

function Stat({ label, value }) {
  return (
    <div
      style={{
        background: 'var(--bg)',
        borderRadius: 'var(--radius-sm)',
        padding: '10px 12px',
      }}
    >
      <div
        style={{
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '.06em',
          color: 'var(--muted)',
          marginBottom: 3,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 13, fontWeight: 500 }}>{value}</div>
    </div>
  )
}