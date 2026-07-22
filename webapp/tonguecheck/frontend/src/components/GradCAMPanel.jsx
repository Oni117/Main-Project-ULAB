import { useState } from 'react'
import s from './GradCAMPanel.module.css'

const VIEWS = [
  {
    key: 'overlay',
    label: 'Overlay',
    desc: 'Heatmap blended onto the original image',
  },
  {
    key: 'heatmap',
    label: 'Grad-CAM',
    desc: 'Activation map — red = most influential regions',
  },
  {
    key: 'original',
    label: 'Original',
    desc: 'Resized input image (240 × 240) fed to the model',
  },
]

function normalizeClassName(value) {
  const cls = String(value || '').toLowerCase().trim()

  if (cls === 'diabetes' || cls === 'diabetic') {
    return {
      display: 'Diabetic',
      isDiabetic: true,
    }
  }

  if (
    cls === 'nondiabetes' ||
    cls === 'non-diabetes' ||
    cls === 'non_diabetes' ||
    cls === 'nondiabetic' ||
    cls === 'non-diabetic'
  ) {
    return {
      display: 'Non-Diabetic',
      isDiabetic: false,
    }
  }

  return {
    display: value || 'Unknown',
    isDiabetic: false,
  }
}

function formatPercent(value) {
  const num = Number(value)

  if (Number.isNaN(num)) return 'N/A'

  const percent = num <= 1 ? num * 100 : num

  return `${percent.toFixed(2)}%`
}

export default function GradCAMPanel({ data, result, loading, error }) {
  const [active, setActive] = useState('overlay')

  const prediction = result?.prediction || result?.pred_class || data?.pred_class
  const confidenceValue = result?.confidence ?? data?.confidence

  const classInfo = normalizeClassName(prediction)
  const confidence = formatPercent(confidenceValue)

  return (
    <div className={s.panel}>
      <div className={s.header}>
        <div className={s.titleRow}>
          <span className={s.title}>Grad-CAM Explanation</span>
          <span className={s.badge}>Explainability</span>
        </div>

        <p className={s.subtitle}>
          Shows which tongue regions influenced the model&apos;s decision most strongly.
        </p>
      </div>

      {loading && (
        <div className={s.center}>
          <span className={s.ring} />
          <p>Generating Grad-CAM…</p>
        </div>
      )}

      {error && !loading && (
        <div className={s.errorBox}>
          <strong>Grad-CAM failed</strong>
          <p>{error}</p>
        </div>
      )}

      {data && !loading && !error && (
        <>
          <div
            className={[
              s.predRow,
              classInfo.isDiabetic ? s.predPos : s.predNeg,
            ].join(' ')}
          >
            <div className={s.predLeft}>
              <span
                className={[
                  s.predBadge,
                  classInfo.isDiabetic ? s.predBadgePos : s.predBadgeNeg,
                ].join(' ')}
              >
                <span className={s.predDot} />
                {classInfo.display}
              </span>

              <span className={s.predConf}>
                {confidence} confidence
              </span>
            </div>

            <span className={s.predLabel}>
              {classInfo.isDiabetic
                ? 'Diabetic indicators detected'
                : 'No diabetic indicators'}
            </span>
          </div>

          <div className={s.tabs}>
            {VIEWS.map((v) => (
              <button
                key={v.key}
                className={[s.tab, active === v.key && s.tabActive]
                  .filter(Boolean)
                  .join(' ')}
                onClick={() => setActive(v.key)}
              >
                {v.label}
              </button>
            ))}
          </div>

          {VIEWS.filter((v) => v.key === active).map((v) => (
            <div key={v.key} className={s.imgWrap}>
              <img src={data[v.key]} alt={v.label} className={s.img} />
              <p className={s.imgCaption}>{v.desc}</p>
            </div>
          ))}

          <div className={s.strip}>
            {VIEWS.map((v) => (
              <button
                key={v.key}
                className={[s.thumb, active === v.key && s.thumbActive]
                  .filter(Boolean)
                  .join(' ')}
                onClick={() => setActive(v.key)}
              >
                <img src={data[v.key]} alt={v.label} className={s.thumbImg} />
                <span className={s.thumbLabel}>{v.label}</span>
              </button>
            ))}
          </div>

          <div className={s.legend}>
            <div className={s.legendBar} />
            <div className={s.legendLabels}>
              <span>Low activation</span>
              <span>High activation</span>
            </div>
            <p className={s.legendNote}>
              Red / warm areas were most important to the model&apos;s decision.
              Blue / cool areas had little influence on the prediction.
            </p>
          </div>
        </>
      )}

      {!data && !loading && !error && (
        <div className={s.empty}>
          <p>Grad-CAM will appear here automatically after you run an analysis.</p>
        </div>
      )}
    </div>
  )
}