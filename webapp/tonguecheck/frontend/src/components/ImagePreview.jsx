import s from './ImagePreview.module.css'

export default function ImagePreview({ file, src, onRemove, disabled }) {
  return (
    <div className={s.wrap}>
      <img src={src} alt="Tongue preview" className={s.img} />
      <div className={s.bar}>
        <div className={s.meta}>
          <span className={s.name}>{file.name}</span>
          <span className={s.size}>{(file.size / 1024).toFixed(0)} KB</span>
        </div>
        <button className={s.remove} onClick={onRemove} disabled={disabled}>Remove</button>
      </div>
    </div>
  )
}
