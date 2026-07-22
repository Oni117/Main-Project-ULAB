const API_BASE_URL = 'http://127.0.0.1:8000'

export async function analyzeImage(file, modelName = 'efficientnet_b1_cbam') {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(
    `${API_BASE_URL}/predict?model_name=${encodeURIComponent(modelName)}`,
    {
      method: 'POST',
      body: form,
    }
  )

  if (!res.ok) {
    let msg = `Server error ${res.status}`
    try {
      const data = await res.json()
      msg = data.detail || data.error || msg
    } catch (_) {}
    throw new Error(msg)
  }

  return res.json()
}

export async function fetchGradCAM(file, modelName = 'efficientnet_b1_cbam') {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(
    `${API_BASE_URL}/gradcam?model_name=${encodeURIComponent(modelName)}`,
    {
      method: 'POST',
      body: form,
    }
  )

  if (!res.ok) {
    let msg = `Grad-CAM error ${res.status}`
    try {
      const data = await res.json()
      msg = data.detail || data.error || msg
    } catch (_) {}
    throw new Error(msg)
  }

  // Grad-CAM returns JSON: { original, heatmap, overlay, pred_class, confidence }
  return res.json()
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/health`)

  if (!res.ok) {
    throw new Error('API unreachable')
  }

  return res.json()
}