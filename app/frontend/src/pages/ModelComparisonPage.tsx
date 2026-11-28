import { useRef, useState } from 'react'
import {
  ArrowRight, CheckCircle2, Clock3, ImagePlus, LoaderCircle,
  ScanLine, ShieldCheck, UploadCloud, XCircle,
} from 'lucide-react'
import { motion } from 'motion/react'
import { predictImage } from '../services/api'
import type { ModelName, PredictionResponse } from '../types/api'

type ComparisonResult = {
  model: ModelName
  response: PredictionResponse
}

export default function ModelComparisonPage() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [results, setResults] = useState<ComparisonResult[]>([])
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [error, setError] = useState('')

  function selectImage(selected?: File) {
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setError('Please select a valid image.')
      return
    }

    setFile(selected)
    setResults([])
    setError('')
    setProgress('')

    const reader = new FileReader()
    reader.onload = () => setPreview(String(reader.result))
    reader.onerror = () => setError('Unable to read the selected image.')
    reader.readAsDataURL(selected)
  }

  async function compareModels() {
    if (!file) {
      setError('Upload an image before starting comparison.')
      return
    }

    setRunning(true)
    setError('')
    setResults([])

    const models: ModelName[] = ['unet', 'yolo11-seg']

    try {
      for (const model of models) {
        setProgress(`Running ${model === 'unet' ? 'U-Net' : 'YOLO11-seg'}...`)
        const response = await predictImage(file, model)
        setResults(previous => [...previous, { model, response }])
      }
      setProgress('Both model predictions completed.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed.')
    } finally {
      setRunning(false)
    }
  }

  return (
    <motion.div className="page-content feature-page comparison-page"
      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <section className="feature-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> EXPERIMENT / MODEL EVALUATION</div>
          <h1>Model comparison<span>.</span></h1>
          <p>Run both trained segmentation models on the same image and inspect their outputs side by side.</p>
        </div>
        <div className="feature-heading-icon"><ScanLine size={22} /></div>
      </section>

      <section className="panel comparison-upload">
        <div>
          <div className="eyebrow">EXPERIMENT INPUT</div>
          <h2>Use one image for both models</h2>
          <p>This experiment runs U-Net and YOLO11-seg sequentially using the same input image.</p>
        </div>

        <input ref={inputRef} type="file" accept="image/*" hidden
          onChange={event => selectImage(event.target.files?.[0])} />

        <button type="button" className="comparison-drop"
          onClick={() => inputRef.current?.click()}
          onDragOver={event => event.preventDefault()}
          onDrop={event => {
            event.preventDefault()
            selectImage(event.dataTransfer.files[0])
          }}>
          {preview
            ? <img src={preview} alt="Comparison input preview" />
            : <><UploadCloud size={27} /><strong>Drop an image or browse files</strong><span>One image, two model predictions</span></>}
        </button>

        {file && <div className="comparison-file">
          <ImagePlus size={16} /> {file.name}
          <button type="button" onClick={() => {
            setFile(null); setPreview(''); setResults([]); setError('')
          }}><XCircle size={16} /> Remove</button>
        </div>}

        <button type="button" className="run-prediction-button"
          disabled={!file || running} onClick={compareModels}>
          {running
            ? <><LoaderCircle className="spin" size={17} /> {progress}</>
            : <>Run both models <ArrowRight size={17} /></>}
        </button>

        {error && <p className="analysis-error">{error}</p>}
        {progress && !running && <p className="comparison-progress"><CheckCircle2 size={15} /> {progress}</p>}
      </section>

      <div className="comparison-results-heading">
        <div>
          <div className="eyebrow">SIDE-BY-SIDE OUTPUT</div>
          <h2>Prediction results</h2>
        </div>
        <span>{results.length} / 2 completed</span>
      </div>

      <div className="comparison-result-grid">
        {(['unet', 'yolo11-seg'] as ModelName[]).map(model => {
          const result = results.find(item => item.model === model)
          const label = model === 'unet' ? 'U-Net' : 'YOLO11-seg'
          const fire = result?.response.classes.fire
          const smoke = result?.response.classes.smoke

          return (
            <article className="panel comparison-model-card" key={model}>
              <div className="comparison-model-header">
                <div>
                  <div className="eyebrow">SEGMENTATION MODEL</div>
                  <h3>{label}</h3>
                </div>
                {result
                  ? <span className="comparison-complete"><CheckCircle2 size={14} /> Complete</span>
                  : <span className="comparison-pending">Awaiting result</span>}
              </div>

              {result ? <>
                <div className="comparison-output-image">
                  <img src={`/api/v1/results/${result.response.result_id}/overlay`}
                    alt={`${label} segmentation overlay`} />
                </div>
                <div className="comparison-live-metrics">
                  <div><Clock3 size={15} /><span>Inference</span><strong>{result.response.processing_time_seconds.toFixed(4)} s</strong></div>
                  <div><ShieldCheck size={15} /><span>Image size</span><strong>{result.response.image.width} × {result.response.image.height}</strong></div>
                  <div><span>Fire coverage</span><strong>{fire?.coverage_percent.toFixed(3)}%</strong></div>
                  <div><span>Smoke coverage</span><strong>{smoke?.coverage_percent.toFixed(3)}%</strong></div>
                </div>
                <div className="comparison-downloads">
                  <a href={`/api/v1/results/${result.response.result_id}/overlay`} target="_blank" rel="noreferrer">View overlay</a>
                  <a href={`/api/v1/results/${result.response.result_id}/fire-mask`} target="_blank" rel="noreferrer">Fire mask</a>
                  <a href={`/api/v1/results/${result.response.result_id}/smoke-mask`} target="_blank" rel="noreferrer">Smoke mask</a>
                </div>
              </> : <div className="comparison-empty">
                <ScanLine size={25} />
                <span>Run the experiment to see {label} output.</span>
              </div>}
            </article>
          )
        })}
      </div>

      <article className="panel comparison-method-note">
        <ShieldCheck size={19} />
        <div>
          <strong>How to interpret this comparison</strong>
          <p>Inference time and predicted class coverage are measured for each output. These values do not establish which prediction is more accurate. Dice and IoU require a verified ground-truth mask for the evaluated image. Use the Phase 05 held-out benchmark for the existing verified metric comparison.</p>
        </div>
      </article>
    </motion.div>
  )
}
