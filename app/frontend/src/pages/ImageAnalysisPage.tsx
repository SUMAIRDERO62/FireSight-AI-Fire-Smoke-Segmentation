import { useRef, useState } from 'react'
import {
  AlertCircle, ArrowRight, CheckCircle2, FileImage, Layers3,
  ImagePlus, LoaderCircle, ScanLine, UploadCloud,
} from 'lucide-react'
import { motion } from 'motion/react'
import ResultPanel from '../components/ResultPanel'
import { predictImage } from '../services/api'
import type { ModelName, PredictionRecord } from '../types/api'

type Props = {
  onPredictionComplete: (record: PredictionRecord) => void
  latestResult: PredictionRecord | null
}

export default function ImageAnalysisPage({ onPredictionComplete, latestResult }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [model, setModel] = useState<ModelName>('yolo11-seg')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  function selectFile(selected: File | undefined) {
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setError('Please select a valid image file.')
      return
    }

    setError('')
    setFile(selected)

    const reader = new FileReader()
    reader.onload = () => setPreview(String(reader.result))
    reader.onerror = () => setError('Unable to read this image.')
    reader.readAsDataURL(selected)
  }

  async function runPrediction() {
    if (!file) {
      setError('Please upload an image first.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await predictImage(file, model)
      const record: PredictionRecord = {
        response,
        originalImage: preview,
        fileName: file.name,
      }
      onPredictionComplete(record)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div className="page-content feature-page" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <section className="feature-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> COMPUTER VISION / INFERENCE</div>
          <h1>Image analysis<span>.</span></h1>
          <p>Upload an image and run fire & smoke segmentation with a trained model.</p>
        </div>
        <div className="feature-heading-icon"><ScanLine size={22} /></div>
      </section>

      <div className="analysis-grid">
        <section className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">INPUT IMAGE</div>
              <h2>Upload an image</h2>
              <p>Choose a clear image containing a scene you want to analyze.</p>
            </div>
          </div>

          <button
            type="button"
            className={`drop-zone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              selectFile(event.dataTransfer.files[0])
            }}
          >
            {preview ? (
              <img className="upload-preview" src={preview} alt="Selected upload preview" />
            ) : (
              <>
                <span className="upload-icon"><UploadCloud size={27} /></span>
                <strong>Drop your image here</strong>
                <span>or click to browse files</span>
                <small>Image formats supported by the API</small>
              </>
            )}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(event) => selectFile(event.target.files?.[0])}
          />

          {file && (
            <div className="selected-file">
              <FileImage size={17} />
              <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
              <button type="button" onClick={() => { setFile(null); setPreview(''); setError('') }}>Remove</button>
            </div>
          )}

          <div className="model-select-heading">
            <div className="eyebrow">SEGMENTATION MODEL</div>
            <span>Select one model for inference</span>
          </div>
          <div className="model-select-grid">
            <button type="button" className={`model-option ${model === 'unet' ? 'selected' : ''}`} onClick={() => setModel('unet')}>
              <span className="model-option-icon"><Layers3 size={18} /></span>
              <span><strong>U-Net</strong><small>Pixel-level segmentation</small></span>
              {model === 'unet' && <CheckCircle2 className="model-check" size={17} />}
            </button>
            <button type="button" className={`model-option ${model === 'yolo11-seg' ? 'selected' : ''}`} onClick={() => setModel('yolo11-seg')}>
              <span className="model-option-icon amber-option"><ScanLine size={18} /></span>
              <span><strong>YOLO11-seg</strong><small>Instance segmentation</small></span>
              {model === 'yolo11-seg' && <CheckCircle2 className="model-check" size={17} />}
            </button>
          </div>

          {error && <div className="analysis-error"><AlertCircle size={16} /> {error}</div>}

          <button className="run-prediction-button" type="button" disabled={!file || loading} onClick={runPrediction}>
            {loading ? <><LoaderCircle className="spin" size={17} /> Running segmentation…</> : <>Run segmentation <ArrowRight size={17} /></>}
          </button>
          <p className="upload-note">Image is sent to your local FireSight AI backend for processing.</p>
        </section>

        <aside className="analysis-guide">
          <div className="guide-mark"><ImagePlus size={20} /></div>
          <div className="eyebrow">WORKFLOW</div>
          <h2>From image to insight.</h2>
          <p>Run a trained model and inspect its segmentation output.</p>
          <div className="guide-step"><span>01</span><div><strong>Upload image</strong><small>Select a local image file.</small></div></div>
          <div className="guide-step"><span>02</span><div><strong>Choose model</strong><small>U-Net or YOLO11-seg.</small></div></div>
          <div className="guide-step"><span>03</span><div><strong>Inspect output</strong><small>Compare masks and coverage.</small></div></div>
          <div className="guide-note">For research and demonstration purposes. Always verify critical situations independently.</div>
        </aside>
      </div>

      {latestResult && <ResultPanel record={latestResult} />}
    </motion.div>
  )
}



