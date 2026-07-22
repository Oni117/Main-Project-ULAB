import { useRef, useState } from 'react'
import s from './UploadZone.module.css'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

export default function UploadZone({ onFile, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function process(f) {
    if (!f) return
    if (!ACCEPTED.includes(f.type)) { alert('Please upload a JPEG, PNG or WEBP image.'); return }
    if (f.size > 10 * 1024 * 1024)  { alert('Image must be under 10 MB.'); return }
    onFile(f)
  }

  return (
    <div
      className={[s.zone, dragging && s.over, disabled && s.disabled].filter(Boolean).join(' ')}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); process(e.dataTransfer.files[0]) }}
    >
      <input
        ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp"
        style={{ display:'none' }} onChange={e => process(e.target.files[0])}
      />
      <div className={s.icon}>
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <p className={s.primary}>Drop tongue image here</p>
      <p className={s.hint}>or click to browse · JPG · PNG · WEBP · max 10 MB</p>
    </div>
  )
}
